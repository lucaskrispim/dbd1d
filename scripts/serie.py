# SPDX-License-Identifier: GPL-3.0-or-later
"""Observables of the time series of a run.

Usage: python3 scripts/serie.py DIR [DIR ...] [--reference REF | --meshes [--floor P]]

For each run: the first validity warning in the log (DIR.log), from which the
run is not valid, and the energy dissipated per period and per electrode area
from the Lissajous curve, E = closed integral of V dQ, with the electrode charge
Q = eps0 eps_d (V - phi_left) / thickness (robust with a coarse output, unlike
the instantaneous power), and the charge transferred to the left surface in each
half period, the change of sigma_left, which with one pulse per half period is
the charge per pulse and does not depend on resolving the time of the pulse.
With the pulse waveform, only the energy per repetition period; with the step
waveform, no quantity per period.

With --meshes, the directories are a series of meshes (coarse to fine, ratio 2):
order and GCI of the energy of each period and of the charge of each half
period, with the classification of convergencia.py.

With --reference (tolerance study): for each observable, the largest difference
over all common times, divided by the maximum of the reference series, and the
time at which it occurs. Measuring at a single time missed the tolerance error
of the breakdown.
"""
import argparse
import pathlib
import re
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from convergencia import incerteza_eca_hoekstra, ordem_e_gci  # noqa: E402
from resumo import blocos, especies_do_caso, numero, parametros_da_execucao, rotulo  # noqa: E402

EPS0 = 8.8541878128e-12


def energia_por_periodo(t, v, phi_esq, espessura, eps_d, periodo):
    """Closed integral of V dQ (J/m2) in each complete period of the series, by the trapezoidal rule."""
    t, v, phi_esq = (np.asarray(a, dtype=float) for a in (t, v, phi_esq))
    q = EPS0 * eps_d * (v - phi_esq) / espessura
    folga = 1e-6 * periodo
    energias = []
    k = 0
    while (k + 1) * periodo <= t[-1] + folga:
        i = (t >= k * periodo - folga) & (t <= (k + 1) * periodo + folga)
        vi, qi = v[i], q[i]
        energias.append(float(np.sum(0.5 * (vi[1:] + vi[:-1]) * np.diff(qi))))
        k += 1
    return energias


def carga_por_semiperiodo(t, sigma, periodo):
    """Change of sigma (C/m2) in each complete half period of the series, between
    the outputs at the limits of the half period."""
    t, sigma = np.asarray(t, dtype=float), np.asarray(sigma, dtype=float)
    meio = 0.5 * periodo
    folga = 1e-6 * periodo
    cargas = []
    k = 0
    while (k + 1) * meio <= t[-1] + folga:
        i0, i1 = (int(np.argmin(np.abs(t - c))) for c in (k * meio, (k + 1) * meio))
        if abs(t[i0] - k * meio) > folga or abs(t[i1] - (k + 1) * meio) > folga:
            raise SystemExit(f"the output lacks the time {k * meio:.4e} or {(k + 1) * meio:.4e} s")
        cargas.append(float(sigma[i1] - sigma[i0]))
        k += 1
    return cargas


def diferenca_maxima(t, obs, t_ref, obs_ref):
    """{name: (largest |f - f_ref| / max |f_ref|, time)} at the common times."""
    t, t_ref = np.asarray(t, dtype=float), np.asarray(t_ref, dtype=float)
    tol = 1e-9 * max(np.abs(t).max(), np.abs(t_ref).max(), 1e-300)
    pares = [(i, j) for i, a in enumerate(t) for j in np.nonzero(np.abs(t_ref - a) <= tol)[0][:1]]
    saida = {}
    for nome, ref in obs_ref.items():
        if nome not in obs:
            continue
        escala = np.abs(ref).max()
        if escala == 0.0 or not pares:
            saida[nome] = (float("nan"), float("nan"))
            continue
        d = [abs(obs[nome][i] - ref[j]) / escala for i, j in pares]
        k = int(np.argmax(d))
        saida[nome] = (float(d[k]), float(t[pares[k][0]]))
    return saida


# only the validity warnings (ionization degree, hot gas); the one on the energy
# outside the table is a diagnostic. Old logs are in Portuguese (AVISO).
AVISO = re.compile(r"(?:AVISO:\s*(?:grau de ionizacao|gas acima)|WARNING:\s*(?:ionization degree|gas above))"
                   r".*?\b(?:t|em t)\s*=\s*([-+0-9.eEdD]+)\s*s")


def primeiro_aviso(log):
    """Time (s) of the first validity warning in the log; None if there is none."""
    log = pathlib.Path(log)
    if not log.exists():
        return None
    tempos = [numero(m.group(1)) for m in map(AVISO.search, log.read_text().splitlines()) if m]
    return min(tempos) if tempos else None


def series(d):
    """Times and observables: the whole superficie.dat and, per block of the
    profiles, n_e and each species of the case (except the background) integrated
    over the gap, and the maximum and integrated temperature rise over the wall
    temperature."""
    sup = np.loadtxt(d / "superficie.dat", skiprows=1, ndmin=2)
    t_sup = sup[:, 0]
    obs_sup = {"V (V)": sup[:, 1], "phi_esq (V)": sup[:, 2], "phi_dir (V)": sup[:, 3],
               "sigma_esq (C/m2)": sup[:, 4], "sigma_dir (C/m2)": sup[:, 5]}
    t_ref = parametros_da_execucao(d).get("t_parede", 300.0)
    especies = especies_do_caso(d)[0]
    t_perf, obs_perf = [], {}
    for tb, nomes, linhas in blocos(d / "perfil.dat"):
        x = [l[0] for l in linhas]
        faces = [x[0]]
        for i in range(1, len(linhas) - 1):
            faces.append(2.0 * x[i] - faces[-1])
        h = np.diff(faces)
        interior = np.array(linhas[1:-1])
        v = {"int ne (cm-3 m)": float(np.sum(interior[:, nomes.index("E")] * h))}
        for coluna in especies:
            if coluna in nomes:
                v[f"int {rotulo(coluna)} (cm-3 m)"] = float(np.sum(interior[:, nomes.index(coluna)] * h))
        if "Temperature" in nomes:
            dt = interior[:, nomes.index("Temperature")] - t_ref
            v["max dT (K)"] = float(dt.max())
            v["int dT (K m)"] = float(np.sum(dt * h))
        t_perf.append(tb)
        for nome, valor in v.items():
            obs_perf.setdefault(nome, []).append(valor)
    return (t_sup, obs_sup), (np.array(t_perf), {k: np.array(v) for k, v in obs_perf.items()})


def log_da_execucao(d):
    for log in (d.parent / (d.name + ".log"), d / "log.txt"):
        if log.exists():
            return log
    return d.parent / (d.name + ".log")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("dirs", nargs="+", type=pathlib.Path)
    g = p.add_mutually_exclusive_group()
    g.add_argument("--reference", "--referencia", dest="referencia", type=pathlib.Path)
    g.add_argument("--meshes", "--malhas", dest="malhas", action="store_true",
                   help="the directories are a series of meshes, ratio 2")
    p.add_argument("--floor", "--piso", dest="piso", type=float, default=0.0,
                   help="with --meshes, measured relative tolerance error; differences below it give no order")
    a = p.parse_args()
    por_periodo = []
    ref = series(a.referencia) if a.referencia else None
    for d in a.dirs:
        print(f"== {d}")
        aviso = primeiro_aviso(log_da_execucao(d))
        print("no warning" if aviso is None else f"first warning at t = {aviso:.4e} s: not valid from there on")
        par = parametros_da_execucao(d)
        (t_sup, sup), perf = series(d)
        forma = par.get("forma", "seno")
        if par.get("espessura", 0.0) > 0.0 and forma == "degrau":
            print("no quantities per period (step waveform)")
        elif par.get("espessura", 0.0) > 0.0:
            # with pulses, the period is the repetition period and there are no half periods of polarity
            e = energia_por_periodo(t_sup, sup["V (V)"], sup["phi_esq (V)"], par["espessura"],
                                    par.get("eps_dieletrico", 4.8), par.get("periodo", 1e-7))
            print("energy per period (J/m2): " + "  ".join(f"{x:.5e}" for x in e))
            metricas = {f"energy period {k + 1}": x for k, x in enumerate(e)}
            if forma == "seno":
                c = carga_por_semiperiodo(t_sup, sup["sigma_esq (C/m2)"], par.get("periodo", 1e-7))
                print("charge per half period (C/m2): " + "  ".join(f"{x:+.5e}" for x in c))
                metricas.update({f"charge half period {k + 1}": x for k, x in enumerate(c)})
            por_periodo.append(metricas)
        if ref:
            print(f"largest difference from {a.referencia.name} (relative to the maximum of the series):")
            for (t, obs), (t_r, obs_r) in zip(((t_sup, sup), perf), ref):
                for nome, (dif, tm) in diferenca_maxima(t, obs, t_r, obs_r).items():
                    print(f"  {nome:20s} {dif:10.3e}  at t = {tm:.4e} s")

    if a.malhas:
        if len(por_periodo) != len(a.dirs) or len(a.dirs) < 3:
            sys.exit("--meshes needs at least three runs with dielectric and a sine or pulse waveform; "
                     "with a step, use convergencia.py per time")
        print("convergence per period (coarse -> fine): orders, classification, fine GCI, "
              "least-squares uncertainty (Eca and Hoekstra)")
        for nome in por_periodo[-1]:
            f = [v.get(nome, float("nan")) for v in por_periodo]
            r = ordem_e_gci(f, 2.0, piso=a.piso)
            eh = incerteza_eca_hoekstra(f, 2.0 ** -np.arange(len(f)))
            rel = eh["U"] / abs(f[-1]) if f[-1] != 0 else float("nan")
            print(f"  {nome:22s} " + "  ".join(f"{o:5.2f}" for o in r["ordens"])
                  + f"   {r['tipo']}   {r['gci']:9.2e}   U (E-H): {rel:9.2e} ({eh['ajuste']}, p {eh['p']:.2f})")


if __name__ == "__main__":
    main()
