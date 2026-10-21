# SPDX-License-Identifier: GPL-3.0-or-later
"""Observed order of convergence and GCI of a refinement series.

Usage: python3 -I scripts/convergencia.py --t T --ratio R DIR_COARSE DIR_MEDIUM DIR_FINE [DIR ...]

The directories are runs that differ only in the mesh (or only in dt), from the
coarsest to the finest, with a constant refinement ratio R between neighbours.
For each observable at time T (s), computes the observed order
    p = ln(|f1 - f2| / |f2 - f3|) / ln(R)
in each consecutive triplet, the classification of the last triplet (monotonic,
oscillatory, divergent, below the tolerance floor given by --floor) and, only
for a monotonic one, Roache's grid convergence index on the finest solution,
GCI = Fs * |f2 - f3| / |f3| / (R**min(p, pf) - 1), with Fs = 1.25 if p is within
10% of the formal order pf (--formal-order) and Fs = 3 otherwise.

Observables of an ionization front (with --front): the Ar+ peak near the cathode
is a structure of ~0.65 um that converges only with cells below 0.5 um, so the
series also measures quantities that do not depend on resolving it: the Ar+
plateau behind the front (median between 3 and 33 um behind the peak), Ar+
integrated over a fixed window of 60 um centred on the peak of the finest run,
and the position of the front (edge of the region with n_e > 1e12 cm-3 on the
cathode side).

Observables: surface charge and potential at both interfaces (superficie.dat);
maximum and gap integral of the density of electrons and of the main ion (Ar+ in
argon), gap integral of each species of the case except the background gas
(kinet.inp and gas.json of the case recorded in parametros.json), E/N at the
centre and, with heating, the maximum and integrated temperature rise over the
wall temperature (perfil.dat). The integrals use the positions of each mesh, so
they do not depend on the number of cells.
"""
import argparse
import math
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from resumo import blocos, especies_do_caso, numero, parametros_da_execucao, rotulo  # noqa: E402


def superficie(diretorio, t):
    linhas = (diretorio / "superficie.dat").read_text().splitlines()[1:]
    for linha in linhas:
        v = [numero(c) for c in linha.split()]
        if abs(v[0] - t) <= 1e-6 * t:
            return {"sigma_esq (C/m2)": v[4], "sigma_dir (C/m2)": v[5],
                    "phi_esq (V)": v[2], "phi_dir (V)": v[3]}
    raise SystemExit(f"{diretorio}: time {t} not found in superficie.dat")


def perfil(diretorio, t):
    for tb, nomes, linhas in blocos(diretorio / "perfil.dat"):
        if abs(tb - t) <= 1e-6 * t:
            especies, ion = especies_do_caso(diretorio)
            jx, jn, ji, je = (nomes.index(c) for c in ("Pos", "E", ion, "E/N"))
            t_ref = parametros_da_execucao(diretorio).get("t_parede", 300.0)
            x = [l[jx] for l in linhas]
            interior = range(1, len(linhas) - 1)

            # faces of the cells: each centre lies midway between its two faces,
            # which also holds on the clustered mesh
            faces = [x[0]]
            for i in interior:
                faces.append(2.0 * x[i] - faces[-1])

            def integral(j):
                return sum(linhas[i][j] * (faces[i] - faces[i - 1]) for i in interior)

            meio = min(interior, key=lambda i: abs(x[i] - 0.5 * x[-1]))
            celulas = (np.array(x[1:-1]), np.diff(faces),
                       np.array([linhas[i][jn] for i in interior]),
                       np.array([linhas[i][ji] for i in interior]))
            v = {"max ne (cm-3)": max(linhas[i][jn] for i in interior),
                 f"max {rotulo(ion)} (cm-3)": max(linhas[i][ji] for i in interior),
                 "int ne (cm-3 m)": integral(jn),
                 "E/N at centre (Td)": linhas[meio][je]}
            for coluna in especies:
                if coluna in nomes:
                    v[f"int {rotulo(coluna)} (cm-3 m)"] = integral(nomes.index(coluna))
            if "Temperature" in nomes:
                # temperature rise over the wall temperature: the relative T hides
                # the error, since the rise is a few kelvin over 300 K
                jt = nomes.index("Temperature")
                v["max dT (K)"] = max(linhas[i][jt] for i in interior) - t_ref
                v["int dT (K m)"] = sum((linhas[i][jt] - t_ref) * (faces[i] - faces[i - 1]) for i in interior)
            return celulas, v
    raise SystemExit(f"{diretorio}: time {t} not found in perfil.dat")


def observaveis_frente(x, h, ne, ai, x_ref, ion="Ar+"):
    """Plateau of the main ion (Ar+ in argon; name given by ion), ion within 60 um
    around x_ref and position of the front.

    x and h are the centres and widths of the cells (m), ne and ai the densities
    (cm-3). The cathode is on the side of the gap where the ion peak is; behind
    the front is the opposite side."""
    x, h, ne, ai = (np.asarray(v, dtype=float) for v in (x, h, ne, ai))
    k = int(np.argmax(ai))
    lado = 1.0 if x[k] > 0.5 * (x[0] + x[-1]) else -1.0
    atras = lado * (x[k] - x)
    faixa = ai[(atras > 3e-6) & (atras < 33e-6)]
    plato = float(np.median(faixa)) if len(faixa) else float("nan")
    janela = np.abs(x - x_ref) < 30e-6
    acima = x[ne > 1e12]
    if len(acima) == 0:                          # no front (after breakdown)
        frente = float("nan")
    else:
        frente = acima.max() if lado > 0 else acima.min()
    return {f"plateau {ion} (cm-3)": plato,
            f"{ion} in 60 um (cm-3 m)": float(np.sum(ai[janela] * h[janela])),
            "front (m)": frente}


def ordem_e_gci(f, razao, ordem_formal=1.0, piso=0.0):
    """Observed orders of each triplet and GCI on the finest mesh.

    The last triplet is classified by the ratio R = (f3 - f2)/(f2 - f1):
    monotonic with 0 < R < 1, oscillatory with -1 < R < 0, divergent with
    |R| >= 1, undefined without a finite value, below the floor when
    |f3 - f2|/|f3| < piso (the measured tolerance error) and near the floor up
    to 3 times it, since the difference carries the noise of both runs. Only the
    monotonic one has GCI = Fs |f3 - f2|/|f3| / (R**min(p, pf) - 1): Fs = 1.25 if
    p is within 10% of the formal order pf; otherwise Fs = 3. With p above the
    formal order, p is limited to pf (conservative, Roache); with p below, the
    observed p enters and the GCI grows as 1/(R**p - 1). The step in Fs makes the
    GCI jump by 2.4 between p = 1.1 pf and p just above."""
    f = [float(v) for v in f]
    ordens = []
    for k in range(len(f) - 2):
        d1, d2 = abs(f[k] - f[k + 1]), abs(f[k + 1] - f[k + 2])
        ordens.append(math.log(d1 / d2) / math.log(razao) if d1 > 0 and d2 > 0 else float("nan"))
    e21, e32 = f[-2] - f[-3], f[-1] - f[-2]
    rel = abs(e32) / abs(f[-1]) if f[-1] != 0 else float("inf")
    gci = float("nan")
    if not all(math.isfinite(v) for v in f[-3:]):
        tipo = "undefined"
    elif rel < piso or e32 == 0.0:
        tipo = "below the floor"
    elif rel < 3.0 * piso:
        # the difference carries the noise of two runs: the order is not reliable
        tipo = "near the floor"
    elif e21 == 0.0 or abs(e32 / e21) >= 1.0:
        tipo = "divergent"
    elif e32 / e21 < 0.0:
        tipo = "oscillatory"
    else:
        tipo = "monotonic"
        p = ordens[-1]
        fs = 1.25 if abs(p - ordem_formal) <= 0.1 * ordem_formal else 3.0
        gci = fs * rel / (razao ** min(p, ordem_formal) - 1.0)
    return {"ordens": ordens, "tipo": tipo, "gci": gci}


def incerteza_eca_hoekstra(f, h, ordem_formal=1.0):
    """Numerical uncertainty by least squares over all meshes (Eca and Hoekstra,
    J. Comput. Phys. 262, 104, 2014), without the weighted variant.

    f and h go from the coarsest to the finest (relative h); at least 4 meshes
    are needed. Free fit phi = phi0 + alpha h^p and, depending on p:
      0.5 <= p < formal order + 0.1: the free fit, Fs = 1.25;
      p >= formal order + 0.1: the one with the smallest deviation among
        phi0 + a h, phi0 + a h^2 and phi0 + a h + b h^2 (tie: the one with fewer
        parameters), Fs = 3;
      p < 0.5: U = 3 Delta_M, with Delta_M = max - min of the data.
    eps = |phi_fit(h_fine) - phi0|, sigma with n minus the number of parameters,
    residual = f_fine - phi_fit(h_fine) and dphi = Delta_M/(n - 1). With sigma <
    dphi, U = Fs eps + sigma + |residual|; otherwise U = 3 sigma/dphi (eps + sigma
    + |residual|). Constant data give U = 0. Returns p (of the free fit), ajuste
    (free, h1, h2, h12, amplitude or constant), phi0, eps, sigma, residuo, fs and
    U (absolute)."""
    f, h = np.asarray(f, dtype=float), np.asarray(h, dtype=float)
    saida = dict.fromkeys(("p", "phi0", "eps", "sigma", "residuo", "fs", "U"), float("nan"))
    saida["ajuste"] = "none"
    n = len(f)
    if n < 4 or not np.all(np.isfinite(f)):
        return saida
    amplitude = f.max() - f.min()
    if amplitude == 0.0:
        return {**saida, "ajuste": "constant", "phi0": float(f[-1]), "eps": 0.0, "sigma": 0.0,
                "residuo": 0.0, "U": 0.0}

    def ajuste(colunas):
        A = np.column_stack([np.ones(n), *colunas])
        c, *_ = np.linalg.lstsq(A, f, rcond=None)
        res = f - A @ c
        return c, res, float(np.sqrt(np.sum(res ** 2) / (n - A.shape[1]))) if n > A.shape[1] else 0.0

    def quadrados(p):
        return np.sum(ajuste([h ** p])[1] ** 2)

    grade = np.linspace(0.05, 6.0, 1191)
    k = int(np.argmin([quadrados(p) for p in grade]))
    lo, hi = grade[max(k - 1, 0)], grade[min(k + 1, len(grade) - 1)]
    for _ in range(60):                          # golden section around the minimum
        m1, m2 = lo + 0.382 * (hi - lo), lo + 0.618 * (hi - lo)
        if quadrados(m1) < quadrados(m2):
            hi = m2
        else:
            lo = m1
    p = 0.5 * (lo + hi)
    saida["p"] = float(p)
    if p < 0.5:
        return {**saida, "ajuste": "amplitude", "eps": float(amplitude), "fs": 3.0, "U": float(3.0 * amplitude)}
    if p < ordem_formal + 0.1:
        c, res, _ = ajuste([h ** p])
        sigma = float(np.sqrt(np.sum(res ** 2) / (n - 3)))      # phi0, alfa e p
        nome, fs, termo = "free", 1.25, c[1] * h[-1] ** p
    else:
        candidatos = []
        for nome_k, cols in (("h1", [h]), ("h2", [h ** 2]), ("h12", [h, h ** 2])):
            ck, rk, sk = ajuste(cols)
            termo_k = sum(a * col[-1] for a, col in zip(ck[1:], cols))
            candidatos.append((sk, nome_k, ck, rk, termo_k))
        menor = min(c_[0] for c_ in candidatos)
        tol = 1e-6 * menor + 1e-14 * np.abs(f).max()
        sigma, nome, c, res, termo = next(c_ for c_ in candidatos if c_[0] <= menor + tol)
        fs = 3.0
    eps, residuo = abs(termo), res[-1]
    dphi = amplitude / (n - 1)
    base = eps + sigma + abs(residuo)
    U = fs * eps + sigma + abs(residuo) if sigma < dphi else 3.0 * sigma / dphi * base
    return {**saida, "ajuste": nome, "phi0": float(c[0]), "eps": float(eps), "sigma": float(sigma),
            "residuo": float(residuo), "fs": fs, "U": float(U)}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--t", type=float, required=True)
    p.add_argument("--ratio", "--razao", dest="razao", type=float, default=2.0)
    p.add_argument("--front", "--frente", dest="frente", action="store_true",
                   help="include the plateau, the main ion in the 60 um window and the position of the front")
    p.add_argument("--formal-order", "--ordem-formal", dest="ordem_formal", type=float, default=1.0,
                   help="formal order of the scheme (1: first-order closure at the walls)")
    p.add_argument("--floor", "--piso", dest="piso", type=float, default=0.0,
                   help="measured relative tolerance error; differences below it give no order")
    p.add_argument("dirs", nargs="+", type=pathlib.Path)
    a = p.parse_args()
    if len(a.dirs) < 3:
        sys.exit("at least three runs are needed")
    valores, celulas = [], []
    for d in a.dirs:
        v = superficie(d, a.t)
        c, vp = perfil(d, a.t)
        v.update(vp)
        valores.append(v)
        celulas.append(c)
    if a.frente:
        x, _, _, ai = celulas[-1]
        x_ref = x[np.argmax(ai)]
        for v, c in zip(valores, celulas):
            v.update(observaveis_frente(*c, x_ref=x_ref, ion=rotulo(especies_do_caso(a.dirs[-1])[1])))
    print(f"t = {a.t:.3e} s, ratio {a.razao}, {len(a.dirs)} runs (coarse -> fine)")
    for nome in valores[0]:
        f = [v[nome] for v in valores]
        texto = "  ".join(f"{x:12.5e}" for x in f)
        r = ordem_e_gci(f, a.razao, a.ordem_formal, a.piso)
        print(f"{nome:20s} {texto}")
        eh = incerteza_eca_hoekstra(f, a.razao ** -np.arange(len(f)), a.ordem_formal)
        rel = eh["U"] / abs(f[-1]) if f[-1] != 0 else float("nan")
        print(f"{'':20s} observed order: " + "  ".join(f"{o:6.2f}" for o in r["ordens"])
              + f"   {r['tipo']}   fine GCI: {r['gci']:9.2e}   U (E-H): {rel:9.2e} ({eh['ajuste']}, p {eh['p']:.2f})")


if __name__ == "__main__":
    main()
