# SPDX-License-Identifier: GPL-3.0-or-later
"""Verification against the Townsend breakdown criterion: bare metal electrodes,
DC voltage, local field approximation (no --energy), linear regime.

Reference: the continuous linear problem that the model solves, with constant
coefficients in the uniform field E = V/d. With s the distance from the cathode,
the electrons drift towards +s with w = mu E and diffuse with D, and ionization
gives nu = k N:
    d/ds (w n - D dn/ds) = nu n  ->  n = sum c_j exp(lambda_j s),
    D lambda^2 - w lambda + nu = 0.
The ions only drift to the cathode, and their flux there is the integral of
nu n. Wall conditions of the model (modelo.fluxos), with the Boltzmann factor of
half a cell equal to 1, its limit for a fine mesh:
    cathode (s = 0): Gamma_e = gamma Gamma_i - a_e n v_th/2
    anode   (s = d): Gamma_e = a_e n (v_th/2 + w)
with a_e = (1 - r)/(1 + r). Breakdown is the zero of the 2x2 determinant in V.
The classical criterion gamma (exp(alpha d) - 1) = 1, alpha = nu/w, is kept as a
physical comparison: the thermal flux at the cathode, which takes back part of
the secondary electrons, and diffusion move it.

Simulation: for each voltage, the inventory of positive ions in the gap grows or
decays as exp(s t) after a transient; the rate s is fitted in the final window
and the zero of s(V) gives the discrete breakdown voltage.

Usage: python3 validacao/townsend.py [--meshes 100 200 400 800] [--pd 10 15 20] [--python PY]
"""
import argparse
import functools
import json
import pathlib
import subprocess
import sys

import numpy as np
from scipy.optimize import brentq

RAIZ = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "prototipo"))

ME = 9.1093837015e-31
T_TRANSITORIO = 1.0e-5   # s: a few ion transit times in a gap of 1 to 2 mm at 100 Torr
KB = 1.380649e-23
TORR = 133.322


def raizes(w, d_e, nu):
    """Spatial exponents lambda_1 < lambda_2 of D l^2 - w l + nu = 0 (1/m)."""
    disc = np.sqrt(w * w - 4.0 * d_e * nu)
    return (w - disc) / (2.0 * d_e), (w + disc) / (2.0 * d_e)


def determinante(w, d_e, nu, v_th, gama, gap, a_e=1.0):
    """Determinant of the wall conditions of the continuous linear problem; zero at
    breakdown. With d_e -> 0 and v_th -> 0, it reduces to the classical criterion."""
    l1, l2 = raizes(w, d_e, nu)
    # modes exp(l1 s) and exp(l2 (s - d)): the second, with a large exponent when
    # diffusion is small, is measured from the anode and does not overflow
    no_catodo = (1.0, np.exp(-l2 * gap))
    no_anodo = (np.exp(l1 * gap), 1.0)
    ions = (nu * np.expm1(l1 * gap) / l1, nu * -np.expm1(-l2 * gap) / l2)   # flux at the cathode per mode
    m = np.zeros((2, 2))
    for j, lam in enumerate((l1, l2)):
        fluxo = w - d_e * lam                       # Gamma_e / n of each mode
        m[0, j] = (fluxo + a_e * 0.5 * v_th) * no_catodo[j] - gama * ions[j]
        m[1, j] = (fluxo - a_e * (0.5 * v_th + w)) * no_anodo[j]
    return np.linalg.det(m)


def classico(alfa, gap, gama):
    return gama * np.expm1(alfa * gap) - 1.0


@functools.lru_cache(maxsize=None)
def modelo(gap, pressao):
    """Model built as in the simulation (argon case, local field)."""
    import simula
    a = simula.cria_parser().parse_args(["/dev/null", "--gap", repr(float(gap)), "--thickness", "0", "--pressure",
                                         repr(float(pressao)), "--cells", "10"])
    return simula.monta_modelo(a, RAIZ, None)


def coeficientes_do_modelo(gap, pressao, v):
    """mu, D, T_e and nu of the model built as in the simulation, at the uniform field."""
    md = modelo(float(gap), float(pressao))
    campo = np.full(md.m, v / gap)
    n = np.zeros((md.n_esp, md.m))
    mob, dif, temp, k_b, e_n, extra = md.coeficientes(campo, n)
    k = md.q.coeficientes(k_b, extra["tg"], temp[md.i_e], e_n)
    ion = md.q.reacoes.index("bolsig:AR->AR^+")
    nu = k[ion, 0] * float(np.ravel(extra["n"])[0])     # 1/s
    return mob[md.i_e, 0], dif[md.i_e, 0], temp[md.i_e, 0], nu, e_n[0]


def ruptura_referencia(gap, pressao, gama, r=0.0, sem_termico=False, sem_difusao=False, v_ini=150.0, v_fim=3000.0):
    """Breakdown voltage of the continuous problem and that of the classical criterion (V)."""
    a_e = (1.0 - r) / (1.0 + r)

    def f(v, cla=False):
        mu, d_e, te, nu, _ = coeficientes_do_modelo(gap, pressao, v)
        w = mu * v / gap
        if cla:
            return classico(nu / w, gap, gama)
        v_th = 0.0 if sem_termico else np.sqrt(8.0 * KB * te / (np.pi * ME))
        return determinante(w, 1e-12 * d_e if sem_difusao else d_e, nu, v_th, gama, gap, a_e)

    vs = np.geomspace(v_ini, v_fim, 60)
    out = []
    for cla in (False, True):
        g = [f(v, cla) for v in vs]
        # below some field the root is complex (w^2 < 4 D nu) and the determinant is NaN
        i = next(i for i in range(len(vs) - 1)
                 if np.isfinite(g[i]) and np.isfinite(g[i + 1]) and np.sign(g[i]) != np.sign(g[i + 1]))
        out.append(brentq(lambda v: f(v, cla), vs[i], vs[i + 1], xtol=1e-3))
    return tuple(out)


def roda(destino, gap, pressao, gama, v, celulas, py, t_end, reflexao=0.0):
    gap, pressao, gama, v = (float(x) for x in (gap, pressao, gama, v))
    cmd = [py, str(RAIZ / "prototipo" / "simula.py"), str(destino), "--gap", repr(gap), "--thickness", "0",
           "--pressure", repr(pressao), "--gamma", repr(gama), "--cells", str(celulas), "--clustering", "3.0",
           "--waveform", "step", "--rise", "1e-9", "--v-peak", repr(v), "--neutral", "--ne0", "1e2",
           "--atol", "1e-14", "--t-end", repr(t_end), "--dt-output", repr(t_end / 100),
           "--electron-reflection", repr(float(reflexao))]
    # above breakdown the current grows up to the nonlinear regime, which metal
    # electrodes do not limit, and the integrator may stop: the part written before counts
    subprocess.run(cmd, capture_output=True, env={"OPENBLAS_NUM_THREADS": "1", "PATH": ""})


def taxa_de_crescimento(destino):
    """Growth rate s (1/s) of the inventory of positive ions, fitted in the linear regime."""
    sys.path.insert(0, str(RAIZ / "scripts"))
    from resumo import blocos
    ts, inv = [], []
    for tb, nomes, linhas in blocos(destino / "perfil.dat"):
        a = np.array(linhas[1:-1])
        faces = [0.0]
        for c in a[:, 0]:
            faces.append(2.0 * c - faces[-1])
        h = np.diff(faces)
        ions = sum(a[:, nomes.index(s)] for s in ("AR^+", "AR2^+"))
        ts.append(tb)
        inv.append(float((ions * h).sum()))
    ts, inv = np.array(ts), np.array(inv)
    # linear regime: after the transient and while the inventory stays below 1e4
    # times the initial one (space charge would distort the field)
    j = (ts >= T_TRANSITORIO) & (inv < 1e4 * inv[0])
    if np.count_nonzero(j) < 5:
        raise RuntimeError(f"{destino}: fewer than 5 records in the linear regime")
    return np.polyfit(ts[j], np.log(inv[j]), 1)[0]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--meshes", "--malhas", dest="malhas", type=int, nargs="+", default=[100, 200, 400, 800])
    p.add_argument("--pd", type=float, nargs="+", default=[10.0, 15.0, 20.0], help="p d in Torr cm")
    p.add_argument("--pressure-torr", "--pressao-torr", dest="pressao_torr", type=float, default=100.0)
    p.add_argument("--gamma", "--gama", dest="gama", type=float, default=0.05)
    p.add_argument("--reflection", "--reflexao", dest="reflexao", type=float, default=0.0,
                   help="electron reflection at the walls")
    p.add_argument("--t-end", type=float, default=3e-5)
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--output", "--saida", dest="saida", type=pathlib.Path, default=RAIZ / "execucoes" / "townsend")
    p.add_argument("--reference-only", "--so-referencia", dest="so_referencia", action="store_true")
    a = p.parse_args()
    pressao = a.pressao_torr * TORR
    res = []
    for pd in a.pd:
        gap = pd / a.pressao_torr * 1e-2
        v_ref, v_cla = ruptura_referencia(gap, pressao, a.gama, r=a.reflexao)
        linha = {"pd": pd, "gap": gap, "gama": a.gama, "reflexao": a.reflexao, "V_referencia": v_ref,
                 "V_classico": v_cla, "malhas": {}}
        print(f"p d = {pd} Torr cm: reference {v_ref:.1f} V, classical {v_cla:.1f} V", flush=True)
        if not a.so_referencia:
            for m in a.malhas:
                vs = v_ref * np.array([0.97, 0.99, 1.01, 1.03])
                taxas = []
                for v in vs:
                    d = a.saida / f"g{a.gama:g}_r{a.reflexao:g}_pd{pd:g}_n{m}_V{v:.1f}"
                    roda(d, gap, pressao, a.gama, v, m, a.python, a.t_end, a.reflexao)
                    taxas.append(taxa_de_crescimento(d))
                c = np.polyfit(vs, taxas, 1)
                v_b = -c[1] / c[0]
                linha["malhas"][m] = {"V": v_b, "tensoes": vs.tolist(), "taxas": taxas}
                print(f"  {m} cells: {v_b:.1f} V ({100 * (v_b / v_ref - 1):+.2f} % from the reference)", flush=True)
        res.append(linha)
    a.saida.mkdir(parents=True, exist_ok=True)
    (a.saida / f"townsend_g{a.gama:g}_r{a.reflexao:g}.json").write_text(json.dumps(res, indent=2) + "\n")


if __name__ == "__main__":
    main()
