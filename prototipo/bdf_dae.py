# SPDX-License-Identifier: GPL-3.0-or-later
"""BDF do scipy com matriz de massa diagonal (0 ou 1): resolve M y' = f(t, y),
uma DAE de índice 1 em que as componentes com M = 0 são algébricas (aqui, o
potencial do Poisson). É o _step_impl do scipy 1.15 com duas mudanças:
  LU = lu(M - c J)   e   dy = solve(c f - M (psi + d)).
As componentes algébricas devem ficar fora do controle de erro (atol grande).
Usa partes internas do scipy (testado com 1.15 e 1.18); NEWTON_MAXITER fica no
módulo para poder ser aumentado.
"""
import numpy as np
from scipy.integrate._ivp.bdf import BDF, MAX_ORDER, MIN_FACTOR, MAX_FACTOR, change_D
from scipy.integrate._ivp.common import norm
from scipy.sparse import issparse, diags
from scipy.sparse.linalg import splu
from scipy.linalg import lu_factor, lu_solve

# iterações de Newton por passo (o scipy usa 4; com 8 o pulso de corrente de
# Saifutdinov precisa de um terço menos jacobianos, medido pelo especialista)
NEWTON_MAXITER = 8


def solve_bdf_system_massa(fun, t_new, y_predict, c, psi, LU, solve_lu, scale, tol, massa):
    d = 0
    y = y_predict.copy()
    dy_norm_old = None
    converged = False
    for k in range(NEWTON_MAXITER):
        f = fun(t_new, y)
        if not np.all(np.isfinite(f)):
            break
        dy = solve_lu(LU, c * f - massa * (psi + d))
        dy_norm = norm(dy / scale)
        rate = None if dy_norm_old is None else dy_norm / dy_norm_old
        if (rate is not None and (rate >= 1 or
                rate ** (NEWTON_MAXITER - k) / (1 - rate) * dy_norm > tol)):
            break
        y += dy
        d += dy
        if (dy_norm == 0 or
                rate is not None and rate / (1 - rate) * dy_norm < tol):
            converged = True
            break
        dy_norm_old = dy_norm
    return converged, k + 1, y, d


class BDFMassa(BDF):
    def __init__(self, fun, t0, y0, t_bound, massa, **kw):
        super().__init__(fun, t0, y0, t_bound, **kw)
        self.massa = np.asarray(massa, dtype=float)
        if issparse(self.I):
            self.M = diags(self.massa, format="csc")
        else:
            self.M = np.diag(self.massa)
        # as componentes algébricas não têm derivada: o predictor começa constante
        self.D[1] *= self.massa
        self.contador = {"newton": 0, "falhas": 0, "rejeicoes": 0, "cortes": 0}
        # equilibração por linha antes da LU: as linhas algébricas (c*A, ~1e-7) e as
        # diferenciais (1 - c*J, até 1e12) têm escalas muito diferentes e a LU
        # esparsa acusava matriz singular por cancelamento
        if issparse(self.I):
            def lu(A):
                self.nlu += 1
                A = A.tocsr()
                r = 1.0 / np.maximum(abs(A).max(axis=1).toarray().ravel(), 1e-300)
                return splu(diags(r, format="csc") @ A.tocsc()), r

            def solve_lu(LU, b):
                return LU[0].solve(LU[1] * b)
        else:
            def lu(A):
                self.nlu += 1
                r = 1.0 / np.maximum(np.abs(A).max(axis=1), 1e-300)
                return lu_factor(r[:, None] * A, overwrite_a=True), r

            def solve_lu(LU, b):
                return lu_solve(LU[0], LU[1] * b)
        self.lu = lu
        self.solve_lu = solve_lu

    def _step_impl(self):
        t = self.t
        D = self.D
        max_step = self.max_step
        min_step = 10 * np.abs(np.nextafter(t, self.direction * np.inf) - t)
        if self.h_abs > max_step:
            h_abs = max_step
            change_D(D, self.order, max_step / self.h_abs)
            self.n_equal_steps = 0
        elif self.h_abs < min_step:
            h_abs = min_step
            change_D(D, self.order, min_step / self.h_abs)
            self.n_equal_steps = 0
        else:
            h_abs = self.h_abs
        atol, rtol, order = self.atol, self.rtol, self.order
        alpha, gamma, error_const = self.alpha, self.gamma, self.error_const
        J, LU = self.J, self.LU
        current_jac = self.jac is None
        step_accepted = False
        while not step_accepted:
            if h_abs < min_step:
                return False, self.TOO_SMALL_STEP
            h = h_abs * self.direction
            t_new = t + h
            if self.direction * (t_new - self.t_bound) > 0:
                t_new = self.t_bound
                change_D(D, order, np.abs(t_new - t) / h_abs)
                self.n_equal_steps = 0
                LU = None
            h = t_new - t
            h_abs = np.abs(h)
            y_predict = np.sum(D[:order + 1], axis=0)
            scale = atol + rtol * np.abs(y_predict)
            psi = np.dot(D[1: order + 1].T, gamma[1: order + 1]) / alpha[order]
            converged = False
            c = h / alpha[order]
            while not converged:
                if LU is None:
                    LU = self.lu(self.M - c * J)
                converged, n_iter, y_new, d = solve_bdf_system_massa(
                    self.fun, t_new, y_predict, c, psi, LU, self.solve_lu,
                    scale, self.newton_tol, self.massa)
                self.contador["newton"] += 1
                if not converged:
                    self.contador["falhas"] += 1
                    if current_jac:
                        break
                    J = self.jac(t_new, y_predict)
                    LU = None
                    current_jac = True
            if not converged:
                # Newton não convergiu nem com o jacobiano atual: passo pela metade
                self.contador["cortes"] += 1
                factor = 0.5
                h_abs *= factor
                change_D(D, order, factor)
                self.n_equal_steps = 0
                LU = None
                continue
            safety = 0.9 * (2 * NEWTON_MAXITER + 1) / (2 * NEWTON_MAXITER + n_iter)
            scale = atol + rtol * np.abs(y_new)
            error = error_const[order] * d
            error_norm = norm(error / scale)
            if error_norm > 1:
                self.contador["rejeicoes"] += 1
                factor = max(MIN_FACTOR, safety * error_norm ** (-1 / (order + 1)))
                h_abs *= factor
                change_D(D, order, factor)
                self.n_equal_steps = 0
            else:
                step_accepted = True
        self.n_equal_steps += 1
        self.t = t_new
        self.y = y_new
        self.h_abs = h_abs
        self.J = J
        self.LU = LU
        D[order + 2] = d - D[order + 1]
        D[order + 1] = d
        for i in reversed(range(order + 1)):
            D[i] += D[i + 1]
        if self.n_equal_steps < order + 1:
            return True, None
        if order > 1:
            error_m = error_const[order - 1] * D[order]
            error_m_norm = norm(error_m / scale)
        else:
            error_m_norm = np.inf
        if order < MAX_ORDER:
            error_p = error_const[order + 1] * D[order + 2]
            error_p_norm = norm(error_p / scale)
        else:
            error_p_norm = np.inf
        error_norms = np.array([error_m_norm, error_norm, error_p_norm])
        with np.errstate(divide='ignore'):
            factors = error_norms ** (-1 / np.arange(order, order + 3))
        delta_order = np.argmax(factors) - 1
        order += delta_order
        self.order = order
        factor = min(MAX_FACTOR, safety * np.max(factors))
        self.h_abs *= factor
        change_D(D, order, factor)
        self.n_equal_steps = 0
        self.LU = None
        return True, None
