# SPDX-License-Identifier: GPL-3.0-or-later
"""Adaptive 1D mesh with a fixed number of cells (moving mesh).

The new mesh comes from the equidistribution of a cell density
    rho(x) = rho_base(x) + peso*rho_sol(x),
with rho_base that of the reference mesh (clustered at the walls, which resolves
the sheaths even without gradients) and rho_sol proportional to the monitor, the
sum of the normalized gradients of the chosen quantities (ln n_e, ln n_i, field).
Both parts have the same integral (m cells), and the variation of rho between
neighbouring cells is limited, which limits the ratio of neighbouring widths.

The densities go from the old mesh to the new one by linear reconstruction with
limited slope (minmod) and overlap integrals: the total of each quantity is
conserved exactly, positivity is kept and linear profiles are transferred
without error.
"""
import numpy as np


def centros(faces):
    return 0.5 * (faces[1:] + faces[:-1])


def monitor(faces, campos):
    """Sum, per cell, of |dq/dx|/(max q - min q) of the quantities in campos."""
    xc = centros(faces)
    total = np.zeros(len(xc))
    for q in campos:
        q = np.asarray(q, dtype=float)
        amplitude = q.max() - q.min()
        if amplitude <= 0.0:
            continue
        grad = np.gradient(q, xc)
        total += np.abs(grad) / amplitude
    return total


def faces_por_equidistribuicao(faces, mon, base, peso=1.0, razao_max=1.3, h_min=1.0e-7):
    """New faces (same number of cells) from the monitor mon in the cells of the
    mesh faces, combined with the cell density of the base mesh; no cell becomes
    smaller than h_min (m)."""
    m = len(faces) - 1
    h = np.diff(faces)
    xc = centros(faces)
    # density of the base mesh (cells per metre) at the centre of each old cell
    h_base = np.diff(base)
    k = np.clip(np.searchsorted(base, xc, side="right") - 1, 0, m - 1)
    rho = 1.0 / h_base[k]
    integral_mon = np.sum(mon * h)
    if integral_mon > 0.0:
        rho = rho + peso * m * mon / integral_mon
    # maximum density 1/h_min; since the number of cells is fixed, the final
    # scaling lowers rho a little, so the cap has a margin
    # variation limited per new cell. Inside an old cell the density is constant,
    # and the size of the new cells jumps at the boundary between two old ones:
    # there the ratio is at most razao_max. Where an old cell contains only a
    # fraction n < 1 of a new cell, the allowed drop is razao_max**n, otherwise
    # the limit would accumulate inside a single new cell. n = rho*h*m/total.
    for _ in range(6):
        escala = m / np.sum(rho * h)
        n_novas = rho * h * escala
        for i in range(1, m):
            rho[i] = max(rho[i], rho[i - 1] * razao_max ** (-min(1.0, n_novas[i - 1])))
        for i in range(m - 2, -1, -1):
            rho[i] = max(rho[i], rho[i + 1] * razao_max ** (-min(1.0, n_novas[i + 1])))
    for _ in range(20):
        escala = m / np.sum(rho * h)
        excesso = rho * escala > 1.0 / h_min
        if not excesso.any():
            break
        rho[excesso] = 1.0 / (h_min * escala)
    acumulado = np.concatenate([[0.0], np.cumsum(rho * h)])
    alvo = np.linspace(0.0, acumulado[-1], m + 1)
    novas = np.interp(alvo, acumulado, faces)
    novas[0], novas[-1] = faces[0], faces[-1]
    return novas


def _inclinacoes(u, xc):
    """Minmod slopes per cell (k, m)."""
    s = np.zeros_like(u)
    dx = np.diff(xc)
    d = np.diff(u, axis=1) / dx
    esq, dir_ = d[:, :-1], d[:, 1:]
    mm = np.where(esq * dir_ > 0.0, np.sign(esq) * np.minimum(np.abs(esq), np.abs(dir_)), 0.0)
    s[:, 1:-1] = mm
    # end cells: the slope of the only side; the limitation that preserves
    # positivity there is in transfere
    s[:, 0] = d[:, 0]
    s[:, -1] = d[:, -1]
    return s


def transfere(u, velhas, novas):
    """Cell averages on the new mesh from those on the old mesh, u (k, m)."""
    u = np.atleast_2d(np.asarray(u, dtype=float))
    xc = centros(velhas)
    s = _inclinacoes(u, xc)
    # at the ends, the one-sided linear reconstruction must not become negative
    # inside the cell
    for p in (0, -1):
        meia = 0.5 * (velhas[1] - velhas[0]) if p == 0 else 0.5 * (velhas[-1] - velhas[-2])
        excesso = np.abs(s[:, p]) * meia > u[:, p]
        s[excesso, p] = 0.0
    pontos = np.union1d(velhas, novas)
    a, b = pontos[:-1], pontos[1:]
    meio = 0.5 * (a + b)
    iv = np.clip(np.searchsorted(velhas, meio, side="right") - 1, 0, len(velhas) - 2)
    jn = np.clip(np.searchsorted(novas, meio, side="right") - 1, 0, len(novas) - 2)
    c = xc[iv]
    integral = u[:, iv] * (b - a) + s[:, iv] * 0.5 * ((b - c) ** 2 - (a - c) ** 2)
    total = np.zeros((u.shape[0], len(novas) - 1))
    for k in range(u.shape[0]):
        np.add.at(total[k], jn, integral[k])
    return total / np.diff(novas)


def transfere_energia(ne, n_eps, velhas, novas, ne_novo):
    """Energy density on the new mesh from the mean energy weighted by the
    electrons: eps_new = int(n_eps)/int(n_e) over the new cell, with constant
    averages per old cell. Transferring n_e and n_eps with independent limited
    reconstructions would create spurious mean energies at a steep front; this
    way eps_new stays between the neighbouring values of the old mesh. The total
    energy is conserved up to the difference between this average and that of n_e."""
    a = transfere_constante(np.vstack([np.maximum(ne, 0.0), np.maximum(n_eps, 0.0)]), velhas, novas)
    with np.errstate(invalid="ignore", divide="ignore"):
        eps = np.where(a[0] > 0.0, a[1] / a[0], 0.0)
    return np.maximum(ne_novo, 0.0) * eps


def transfere_constante(u, velhas, novas):
    """Conservative transfer with a constant value per cell (first order)."""
    u = np.atleast_2d(np.asarray(u, dtype=float))
    pontos = np.union1d(velhas, novas)
    a, b = pontos[:-1], pontos[1:]
    meio = 0.5 * (a + b)
    iv = np.clip(np.searchsorted(velhas, meio, side="right") - 1, 0, len(velhas) - 2)
    jn = np.clip(np.searchsorted(novas, meio, side="right") - 1, 0, len(novas) - 2)
    total = np.zeros((u.shape[0], len(novas) - 1))
    for k in range(u.shape[0]):
        np.add.at(total[k], jn, u[k, iv] * (b - a))
    return total / np.diff(novas)
