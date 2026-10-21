# SPDX-License-Identifier: GPL-3.0-or-later
"""Observáveis da série temporal de uma execução.

Uso: python3 scripts/serie.py DIR [DIR ...] [--referencia REF | --malhas [--piso P]]

Para cada execução: o primeiro aviso de validade no log (DIR.log da campanha
ou DIR/log.txt do Fortran), a partir do qual a execução não vale, e a energia
dissipada por período e por área do eletrodo pela curva de Lissajous,
E = ∮ V dQ, com a carga do eletrodo Q = eps0 eps_d (V - phi_esq) / espessura
(robusta com a saída de 1 ns, ao contrário da potência instantânea), e a carga
transferida para a superfície esquerda em cada semiperíodo, a variação de
sigma_esq, que no caso D (um pulso por semiperíodo) é a carga por pulso e não
depende de resolver o instante do pulso com a saída de 50 ns. Com --forma
pulso, só a energia por período de repetição; com --forma degrau, nenhuma
métrica por período.

Com --malhas, os diretórios são uma série de malhas (grossa para fina, razão
2): ordem e GCI da energia de cada período e da carga de cada semiperíodo,
com a classificação de convergencia.py.

Com --referencia (estudo de tolerância): para cada observável, a maior
diferença ao longo de todos os instantes comuns, dividida pelo máximo da série
da referência, e o instante em que ela ocorre. Medir num instante só deixou
passar o erro de tolerância da ruptura (ACHADOS, Fortran contra protótipo).
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
    """∮ V dQ (J/m2) em cada período completo da série, pela regra do trapézio."""
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
    """Variação de sigma (C/m2) em cada semiperíodo completo da série, entre as
    saídas nos limites do semiperíodo."""
    t, sigma = np.asarray(t, dtype=float), np.asarray(sigma, dtype=float)
    meio = 0.5 * periodo
    folga = 1e-6 * periodo
    cargas = []
    k = 0
    while (k + 1) * meio <= t[-1] + folga:
        i0, i1 = (int(np.argmin(np.abs(t - c))) for c in (k * meio, (k + 1) * meio))
        if abs(t[i0] - k * meio) > folga or abs(t[i1] - (k + 1) * meio) > folga:
            raise SystemExit(f"a saída não tem o instante {k * meio:.4e} ou {(k + 1) * meio:.4e} s")
        cargas.append(float(sigma[i1] - sigma[i0]))
        k += 1
    return cargas


def diferenca_maxima(t, obs, t_ref, obs_ref):
    """{nome: (maior |f - f_ref| / max |f_ref|, instante)} nos instantes comuns."""
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


# só os avisos de validade (grau de ionização, gás quente); o de energia fora
# da tabela é diagnóstico
AVISO = re.compile(r"AVISO:\s*(?:grau de ionizacao|gas acima).*?\bt\s*=\s*([-+0-9.eEdD]+)\s*s")


def primeiro_aviso(log):
    """Instante (s) do primeiro aviso de validade no log; None se não há."""
    log = pathlib.Path(log)
    if not log.exists():
        return None
    tempos = [numero(m.group(1)) for m in map(AVISO.search, log.read_text().splitlines()) if m]
    return min(tempos) if tempos else None


def series(d):
    """Instantes e observáveis: superficie.dat inteiro e, por bloco do perfil,
    n_e e cada espécie do caso (menos o fundo) integrados e ΔT máximo e integrado sobre a temperatura de parede."""
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
    g.add_argument("--referencia", type=pathlib.Path)
    g.add_argument("--malhas", action="store_true", help="os diretórios são uma série de malhas, razão 2")
    p.add_argument("--piso", type=float, default=0.0,
                   help="com --malhas, erro relativo de tolerância medido; diferenças abaixo dele não dão ordem")
    a = p.parse_args()
    por_periodo = []
    ref = series(a.referencia) if a.referencia else None
    for d in a.dirs:
        print(f"== {d}")
        aviso = primeiro_aviso(log_da_execucao(d))
        print("sem aviso" if aviso is None else f"primeiro aviso em t = {aviso:.4e} s: não vale a partir daí")
        par = parametros_da_execucao(d)
        (t_sup, sup), perf = series(d)
        forma = par.get("forma", "seno")
        if par.get("espessura", 0.0) > 0.0 and forma == "degrau":
            print("sem métricas por período (forma degrau)")
        elif par.get("espessura", 0.0) > 0.0:
            # com pulso, o período é o de repetição e não há semiperíodos de polaridade
            e = energia_por_periodo(t_sup, sup["V (V)"], sup["phi_esq (V)"], par["espessura"],
                                    par.get("eps_dieletrico", 4.8), par.get("periodo", 1e-7))
            print("energia por período (J/m2): " + "  ".join(f"{x:.5e}" for x in e))
            metricas = {f"energia período {k + 1}": x for k, x in enumerate(e)}
            if forma == "seno":
                c = carga_por_semiperiodo(t_sup, sup["sigma_esq (C/m2)"], par.get("periodo", 1e-7))
                print("carga por semiperíodo (C/m2): " + "  ".join(f"{x:+.5e}" for x in c))
                metricas.update({f"carga semiperíodo {k + 1}": x for k, x in enumerate(c)})
            por_periodo.append(metricas)
        if ref:
            print(f"diferença máxima para {a.referencia.name} (relativa ao máximo da série):")
            for (t, obs), (t_r, obs_r) in zip(((t_sup, sup), perf), ref):
                for nome, (dif, tm) in diferenca_maxima(t, obs, t_r, obs_r).items():
                    print(f"  {nome:20s} {dif:10.3e}  em t = {tm:.4e} s")

    if a.malhas:
        if len(por_periodo) != len(a.dirs) or len(a.dirs) < 3:
            sys.exit("--malhas precisa de pelo menos três execuções com dielétrico e forma seno ou pulso; "
                     "com degrau, use convergencia.py por instante")
        print("convergência por período (grossa -> fina): ordens, classificação, GCI fino, "
              "incerteza por mínimos quadrados (Eça e Hoekstra)")
        for nome in por_periodo[-1]:
            f = [v.get(nome, float("nan")) for v in por_periodo]
            r = ordem_e_gci(f, 2.0, piso=a.piso)
            eh = incerteza_eca_hoekstra(f, 2.0 ** -np.arange(len(f)))
            rel = eh["U"] / abs(f[-1]) if f[-1] != 0 else float("nan")
            print(f"  {nome:22s} " + "  ".join(f"{o:5.2f}" for o in r["ordens"])
                  + f"   {r['tipo']}   {r['gci']:9.2e}   U (E-H): {rel:9.2e} ({eh['ajuste']}, p {eh['p']:.2f})")


if __name__ == "__main__":
    main()
