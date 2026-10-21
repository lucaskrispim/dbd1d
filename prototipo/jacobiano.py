# SPDX-License-Identifier: GPL-3.0-or-later
"""Structured Jacobian of the coupled model and differential-algebraic form with
the potential in the state.

1. ModeloRapido.jacobiano(t, y): dense Jacobian with the structure
       J = J_local + S G,
   J_local = d rhs/d y at fixed field (block tridiagonal, by coloured finite
   differences: 3*n_var evaluations), S = d rhs/d field at the faces (8
   evaluations) and G = d field/d(rho, sigma), the linear response of the
   Poisson equation, computed once per mesh. It replaces the N = n_var*m + 2
   evaluations of scipy's finite-difference Jacobian, whose adaptive step
   collapsed during current pulses and turned the Jacobian into round-off noise.

2. SistemaDAE: the potential at the m+2 nodes is in the state, 0 = A phi - b(n,
   sigma, t) (an index-one DAE), with a sparse banded Jacobian, integrated by
   bdf_dae.BDFMassa (zero mass matrix in the rows of the Poisson equation) with
   sparse LU.
"""
import numpy as np
from scipy import sparse
from scipy.linalg import solve_banded

from modelo import EPS0, Q, Modelo

EPS = np.finfo(float).eps


class ModeloRapido(Modelo):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self._prepara_poisson()

    def troca_malha(self, faces):
        super().troca_malha(faces)
        if hasattr(self, "espessura"):
            self._prepara_poisson()

    # ------------------------------------------------------------ Poisson
    def _matriz_poisson(self):
        m, d = self.m, self.d
        n = m + 2
        ab = np.zeros((3, n))
        for_cel = np.arange(1, m + 1)
        g_w = self.eps_gas / d[for_cel - 1]
        g_e = self.eps_gas / d[for_cel]
        ab[1, for_cel] = -(g_w + g_e)
        ab[0, for_cel + 1] = g_e
        ab[2, for_cel - 1] = g_w
        if self.espessura > 0.0:
            c = self.eps_d / self.espessura
            ab[1, 0] = -(c + self.eps_gas / d[0])
            ab[0, 1] = self.eps_gas / d[0]
            ab[1, n - 1] = -(self.eps_d_dir / self.espessura_dir + self.eps_gas / d[-1])
            ab[2, n - 2] = self.eps_gas / d[-1]
        else:
            ab[1, 0] = 1.0
            ab[1, n - 1] = 1.0
        return ab

    def _rhs_poisson(self, rho, sigma_esq, sigma_dir, v):
        m = self.m
        n = m + 2
        rhs = np.zeros(n)
        rhs[1:m + 1] = -rho * self.h / EPS0
        if self.espessura > 0.0:
            c = self.eps_d / self.espessura
            rhs[0] = -c * v - sigma_esq / EPS0
            rhs[n - 1] = -sigma_dir / EPS0
        else:
            rhs[0] = v
        return rhs

    def _prepara_poisson(self):
        """G: field at the m+1 faces per unit of rho in each cell (columns 0..m-1)
        and per unit of sigma_esq and sigma_dir (columns m, m+1)."""
        m = self.m
        n = m + 2
        self.ab = self._matriz_poisson()
        B = np.zeros((n, m + 2))
        B[np.arange(1, m + 1), np.arange(m)] = -self.h / EPS0
        if self.espessura > 0.0:
            B[0, m] = -1.0 / EPS0
            B[n - 1, m + 1] = -1.0 / EPS0
        Phi = solve_banded((1, 1), self.ab, B)
        self.G = -np.diff(Phi, axis=0) / self.d[:, None]        # (m+1, m+2)
        # sparse matrix A of the Poisson equation (for the DAE form)
        ab = self.ab
        self.A_poisson = sparse.diags([ab[2, :-1], ab[1], ab[0, 1:]], [-1, 0, 1], format="csr")

    def campo_de(self, t, y):
        n, s_esq, s_dir = self.separa(y)
        return self.campo(t, n, s_esq, s_dir)[1]

    # ------------------------------------------------------------ Jacobian
    REL_Y = 1.0e-7          # relative finite-difference step in the densities
    REL_E = 1.0e-6          # and in the field (the nonlinearity in E has a scale of ~1e7 V/m)

    def _passo_fd(self, v, f, limiar, rel):
        sinal = np.where(f >= 0.0, 1.0, -1.0)
        return (v + rel * sinal * np.maximum(np.abs(v), limiar)) - v

    def partes_jacobiano(self, t, y, campo, atol=1e-4, limiar_campo=1.0e4):
        """J_local (sparse, N x N, at fixed field) and S = d rhs/d field (sparse, N x (m+1))."""
        m, nv = self.m, self.n_var
        N = nv * m + 2
        f0 = self.rhs_campo(t, y, campo)
        lin, col, val = [], [], []
        hy = self._passo_fd(y, f0, atol, self.REL_Y)
        cel = np.arange(m)
        bloco = m * np.arange(nv)[:, None]
        for v in range(nv):
            for r in range(3):
                cols = v * m + cel[cel % 3 == r]
                yp = y.copy()
                yp[cols] += hy[cols]
                df = (self.rhs_campo(t, yp, campo) - f0)
                for c in cols:
                    i = c - v * m
                    lo, hi = max(i - 1, 0), min(i + 1, m - 1)
                    linhas = (np.arange(lo, hi + 1)[None, :] + bloco).ravel()
                    lin.append(linhas); col.append(np.full(len(linhas), c)); val.append(df[linhas] / hy[c])
                    if (i == 0 or i == m - 1) and (m - 1) % 3 != 0:
                        lin.append([N - 2, N - 1]); col.append([c, c]); val.append(df[N - 2:] / hy[c])
        if (m - 1) % 3 == 0:
            for v in range(nv):
                for i in (0, m - 1):
                    c = v * m + i
                    yp = y.copy()
                    yp[c] += hy[c]
                    df = self.rhs_campo(t, yp, campo) - f0
                    lin.append([N - 2, N - 1]); col.append([c, c]); val.append(df[N - 2:] / hy[c])
        J_loc = sparse.csc_matrix((np.concatenate(val), (np.concatenate(lin), np.concatenate(col))), shape=(N, N))
        hE = self._passo_fd(campo, np.ones_like(campo), limiar_campo, self.REL_E)
        lin, col, val = [], [], []
        grupos = [[0], [1], [m - 1], [m]] + [list(range(2 + r, m - 1, 4)) for r in range(4)]
        for g in grupos:
            g = [j for j in g if 0 <= j <= m]
            if not g:
                continue
            Ep = campo.copy()
            Ep[g] += hE[g]
            df = self.rhs_campo(t, y, Ep) - f0
            for j in g:
                lo, hi = max(j - 2, 0), min(j + 1, m - 1)
                linhas = (np.arange(lo, hi + 1)[None, :] + bloco).ravel()
                lin.append(linhas); col.append(np.full(len(linhas), j)); val.append(df[linhas] / hE[j])
                if j <= 1 or j >= m - 1:
                    lin.append([N - 2, N - 1]); col.append([j, j]); val.append(df[N - 2:] / hE[j])
        S = sparse.csc_matrix((np.concatenate(val), (np.concatenate(lin), np.concatenate(col))), shape=(N, m + 1))
        return J_loc, S

    def jacobiano(self, t, y, atol=1e-4, limiar_campo=1.0e4):
        """Dense Jacobian of rhs: J_local + S G_y."""
        m, N = self.m, self.n_var * self.m + 2
        campo = self.campo_de(t, y)
        J_loc, S = self.partes_jacobiano(t, y, campo, atol, limiar_campo)
        Gy = np.zeros((m + 1, N))
        for k in range(self.n_esp):
            if self.carga[k] != 0.0:
                Gy[:, k * m:(k + 1) * m] = Q * 1e6 * self.carga[k] * self.G[:, :m]
        Gy[:, N - 2] = self.G[:, m]
        Gy[:, N - 1] = self.G[:, m + 1]
        return J_loc.toarray() + S @ Gy


class SistemaDAE:
    """System z = (y, phi): phi at the m+2 nodes, with 0 = A phi - b (rows of zero
    mass in BDFMassa)."""

    def __init__(self, md):
        self.md = md
        self.N = md.n_var * md.m + 2
        self.n_phi = md.m + 2

    def estado(self, t, y):
        n, s_esq, s_dir = self.md.separa(y)
        phi, _ = self.md.campo(t, n, s_esq, s_dir)
        return np.concatenate([y, phi])

    def separa(self, z):
        return z[:self.N], z[self.N:]

    def rhs(self, t, z):
        md = self.md
        y, phi = self.separa(z)
        campo = -np.diff(phi) / md.d
        f = md.rhs_campo(t, y, campo)
        n, s_esq, s_dir = md.separa(y)
        rho = Q * 1e6 * (md.carga @ n)
        b = md._rhs_poisson(rho, s_esq, s_dir, md.tensao(t))
        g = md.A_poisson @ phi - b
        return np.concatenate([f, g])

    def jacobiano(self, t, z, atol=1e-4):
        """Sparse Jacobian: [[J_loc, S dE/dphi], [-db/dy, A]]."""
        md = self.md
        m, N = md.m, self.N
        y, phi = self.separa(z)
        campo = -np.diff(phi) / md.d
        J_loc, S = md.partes_jacobiano(t, y, campo, atol)
        # dE_j/dphi_j = 1/d_j, dE_j/dphi_{j+1} = -1/d_j
        inv = 1.0 / md.d
        dE = sparse.csc_matrix((np.concatenate([inv, -inv]),
                                (np.concatenate([np.arange(m + 1), np.arange(m + 1)]),
                                 np.concatenate([np.arange(m + 1), np.arange(1, m + 2)]))),
                               shape=(m + 1, m + 2))
        lin, col, val = [], [], []
        for k in range(md.n_esp):
            if md.carga[k] != 0.0:
                lin.append(np.arange(1, m + 1)); col.append(k * m + np.arange(m))
                val.append(Q * 1e6 * md.carga[k] * md.h / EPS0)          # -db/dn = +rho h/eps0 per n
        if md.espessura > 0.0:
            lin.append([0, m + 1]); col.append([N - 2, N - 1]); val.append([1.0 / EPS0, 1.0 / EPS0])
        dB = sparse.csc_matrix((np.concatenate(val), (np.concatenate(lin), np.concatenate(col))),
                               shape=(m + 2, N))
        J = sparse.bmat([[J_loc, S @ dE], [dB, md.A_poisson]], format="csc")
        return J
