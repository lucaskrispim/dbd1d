# SPDX-License-Identifier: GPL-3.0-or-later
"""Integrates the coupled model in time with the BDF method of scipy (adaptive
step and order) and writes superficie.dat and perfil.dat in the format of the
original Fortran code.

Usage: python3 dbd1d/simula.py DIR --v-peak 1e3 --t-end 2e-8 [options]
       --restart ORIGIN T  starts from the state saved in ORIGIN at time T (s)

The options have English names; the Portuguese names of earlier versions
(--espessura, --pressao, --caso, ...) are still accepted.
"""
import argparse
import gc
import hashlib
import json
import os
import platform
import shutil
import subprocess
import pathlib
import sys
import time

# one BLAS thread per process: with several, simultaneous processes compete for
# the cores and the LU factorization of the BDF becomes hundreds of times slower
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_var] = "1"

import numpy as np
from scipy.integrate import BDF, solve_ivp

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from bdf_dae import BDFMassa                 # noqa: E402
from jacobiano import ModeloRapido, SistemaDAE  # noqa: E402
from modelo import KB, ErroGas, ErroTransporte  # noqa: E402
from quimica_zdp import ErroQuimica             # noqa: E402
from versao import VERSAO  # noqa: E402
from malha_adaptativa import faces_por_equidistribuicao, monitor, transfere, transfere_energia  # noqa: E402
from quimica_zdp import Quimica              # noqa: E402
from tabela import TabelaEEDF, TabelaEnergiaTg, TabelaTg     # noqa: E402

RAIZ = pathlib.Path(__file__).resolve().parent.parent


def linha(valores):
    # always a space between numbers: three-digit exponents (E-101) take the
    # whole width and would stick to the neighbour
    return "".join(f" {v:24.16E}" for v in valores) + "\n"


def cabecalho(nomes):
    return "".join(f"{n:>24s}" for n in nomes) + "\n"


def nomes_do_perfil(md):
    """Columns of perfil.dat: those of the Fortran output with the same names
    (without the CFL number, which does not apply to an implicit integrator),
    every species in the order of ZDPlasKin, E/N, the field at the faces
    (column "campo", V/m) and, with the energy equation, n_eps (eV cm-3), which
    makes a restart exact."""
    especies = [e.upper() for e in md.q.especies]
    return (["Pos", "Temperature", "volt", "De", "Me", "veloc", "diff T", "rho cv", "alfa", "freq", "Te"]
            + especies + ["E/N", "campo"] + (["n_eps"] if md.energia else []))


def avisa(md, t, y):
    """One-time warnings of an ionization degree above 1e-3 (fixed background
    gas) and of gas above 1000 K (with heating), with the position."""
    for chave, celula, texto in (("ionizacao", md.celula_muito_ionizada(y), "ionization degree above 1e-3"),
                                 ("quente", md.celula_quente(y), f"gas above {md.T_ALTA:.0f} K")):
        if celula is not None and not getattr(md, "avisou_" + chave, False):
            # node in the numbering of the Fortran code (wall = 1, first cell = 2)
            print(f"WARNING: {texto} at t = {t:.4e} s, node {celula + 2} (x = {md.xc[celula] * 1e6:.2f} um)",
                  flush=True)
            setattr(md, "avisou_" + chave, True)


# EEDF written at the nodes of the Fortran code (2, n/2, n-1, with n = cells + 2):
# first cell, cell floor(m/2) counted from 1, and last cell
EEDF_ARQUIVOS = ("eedf_borda0.dat", "eedf_centro.dat", "eedf_borda1.dat")


def celulas_da_eedf(md):
    return (0, (md.m + 2) // 2 - 2, md.m - 1)


def energia_da_eedf(md, e_n, extra):
    """Mean energy of the written EEDF: that of the state (with --energy) or the
    local field value at the E/N and T_g of the cell."""
    if md.energia:
        return extra["eps"]
    if not hasattr(md, "tab_e_lfa"):
        md.tab_e_lfa = TabelaEnergiaTg(md.caso / "tabela_energia_tg.dat")
    return md.tab_e_lfa.energia_de_campo_local(e_n, extra["tg"])


def grava_eedf(dir_, md, t, primeiro, e_n, extra):
    """EEDF of the table in three cells, at the mean energy (with --energy) or at
    the local field value."""
    if not hasattr(md, "tab_eedf"):
        md.tab_eedf = TabelaEEDF(md.caso / "tabela_eedf.dat")
    eps = energia_da_eedf(md, e_n, extra)
    for nome, c in zip(EEDF_ARQUIVOS, celulas_da_eedf(md)):
        u, f = md.tab_eedf(eps[c])
        with open(dir_ / nome, "w" if primeiro else "a") as arq:
            arq.write(f"{t:24.16E}\n" + cabecalho(["Energy", "EEDF"]))
            arq.write("".join(linha([a, b]) for a, b in zip(u, f)) + "\n")


def grava(dir_, md, ts, ys, primeiro):
    """Appends to (or, for the first record, creates) superficie.dat, perfil.dat
    and the EEDF files with the states ys (columns) at the times ts, in the
    format of the Fortran output."""
    modo = "w" if primeiro else "a"
    with open(dir_ / "superficie.dat", modo) as sup, open(dir_ / "perfil.dat", modo) as per:
        if primeiro:
            # column names kept from the Fortran output (scripts/compara.py matches them):
            # applied voltage, potential and surface charge on the left and right barriers
            sup.write(cabecalho(["Time", "V aplicada", "phi esq", "phi dir", "sigma esq", "sigma dir"]))
        nomes = nomes_do_perfil(md)
        for t, y in zip(ts, ys.T):
            n, s_esq, s_dir = md.separa(y)
            phi, campo = md.campo(t, n, s_esq, s_dir)
            campo_cel = md.campo_nas_celulas(campo)
            n_eps = md.energia_do_estado(y) if md.energia else None
            t_gas = md.temperatura_do_estado(y) if md.aquecimento else np.full(md.m, md.t_gas)
            mob, dif, temp, _, e_n, extra = md.coeficientes(campo_cel, n, n_eps,
                                                             t_gas if md.aquecimento else None)
            freq = md.frequencia_de_colisao(n, extra)
            if md.energia:
                acima, abaixo = md.fora_da_tabela(y)
                if acima or abaixo:
                    print(f"WARNING: mean energy outside the table at t = {t:.4e} s: {acima} cells above, "
                          f"{abaixo} below", flush=True)
            avisa(md, t, y)
            grava_eedf(dir_, md, t, primeiro and t == ts[0], e_n, extra)
            kap = md.condutividade(t_gas)
            rho_cv = md.cv_por_kb * KB * md.n_pesadas(n) * 1e6
            # densities of every species; the fixed background gas with its density
            dens = np.empty((len(md.q.especies), md.m))
            dens[md.especies] = n
            if not md.gas_solto:
                for i, fracao in md.fundos:
                    dens[i] = fracao * md.n_gas
            sup.write(linha([t, md.tensao(t), phi[0], phi[-1], s_esq, s_dir]))
            per.write(f"{t:24.16E}\n")
            per.write(cabecalho(nomes))
            x = np.concatenate([[0.0], md.xc, [md.L]])
            t_par = md.t_parede if md.aquecimento else md.t_gas
            k_par = float(md.condutividade(np.array(t_par)))
            for i in range(md.m + 2):
                if 0 < i <= md.m:
                    c = i - 1
                    me = mob[md.i_e, c]
                    valores = [x[i], t_gas[c], phi[i], dif[md.i_e, c], me, me * abs(campo_cel[c]), kap[c],
                               rho_cv[c], kap[c] / rho_cv[c], freq[c], temp[md.i_e, c], *dens[:, c],
                               e_n[c], campo[i]]
                    if md.energia:
                        valores.append(n_eps[c])
                else:
                    # wall nodes, as in the Fortran code: wall temperature and kappa,
                    # rho cv of the neighbouring cell, zero coefficients and densities
                    viz = 0 if i == 0 else md.m - 1
                    valores = [x[i], t_par, phi[i], 0.0, 0.0, 0.0, k_par, rho_cv[viz],
                               k_par / rho_cv[viz], 0.0, 0.0, *np.zeros(len(md.q.especies)), 0.0,
                               campo[min(i, md.m)]]
                    if md.energia:
                        valores.append(0.0)
                per.write(linha(valores))
            per.write("\n")


def readapta(md, y, t, peso):
    """New mesh by equidistribution (ln of the charged densities, with a floor of
    1e-6 of the maximum and never below 1e8 cm-3, densities that do not affect
    the field; and the field in the cells) and the state transferred to it
    conservatively; the electron energy goes through the weighted mean energy.
    The model then uses the new mesh."""
    n, s_esq, s_dir = md.separa(y)
    z = y[:-2].reshape(md.n_var, md.m)
    _, campo = md.campo(t, n, s_esq, s_dir)
    grandezas = [md.campo_nas_celulas(campo)]
    for k in range(md.n_esp):
        if md.carga[k] != 0.0:
            piso = max(1e-6 * n[k].max(), 1.0e8)
            grandezas.append(np.log(np.maximum(n[k], piso)))
    novas = faces_por_equidistribuicao(md.faces, monitor(md.faces, grandezas), md.base, peso)
    z_nova = transfere(z, md.faces, novas)
    if md.energia:
        z_nova[md.n_esp] = transfere_energia(z[md.i_e], z[md.n_esp], md.faces, novas, z_nova[md.i_e])
    md.troca_malha(novas)
    return np.concatenate([z_nova.ravel(), [s_esq, s_dir]])


def estado_gravado(md, origem, t):
    """State y at time t from perfil.dat and superficie.dat. Refuses an origin
    with another mesh; the energy comes from the column n_eps (exact) or, in old
    files, from the written Te."""
    sys.path.insert(0, str(RAIZ / "scripts"))
    from resumo import blocos
    for tb, nomes, linhas in blocos(origem / "perfil.dat"):
        if abs(tb - t) <= 1e-9 * t:
            interior = np.array(linhas[1:-1])
            if len(interior) != md.m or not np.allclose(interior[:, 0], md.xc, rtol=1e-12, atol=0):
                raise SystemExit(f"the mesh of {origem} differs from that of this run")
            faltam = [c for c in md.nomes if c not in nomes]
            if md.aquecimento and "Temperature" not in nomes and "Tgas" not in nomes:
                faltam.append("Temperature")
            if faltam:
                raise SystemExit(f"{origem}/perfil.dat lacks the columns {', '.join(faltam)}")
            n = np.array([interior[:, nomes.index(nome)] for nome in md.nomes])
            if md.energia:
                if "n_eps" in nomes:
                    n_eps = interior[:, nomes.index("n_eps")]
                else:
                    n_eps = n[md.i_e] * 1.5 * 1.380649e-23 * interior[:, nomes.index("Te")] / 1.602176634e-19
                n = np.vstack([n, n_eps])
            if md.aquecimento:
                coluna = "Temperature" if "Temperature" in nomes else "Tgas"
                n = np.vstack([n, interior[:, nomes.index(coluna)]])
            break
    else:
        raise SystemExit(f"time {t} not found in {origem}/perfil.dat")
    for l in (origem / "superficie.dat").read_text().splitlines()[1:]:
        v = [float(x) for x in l.split()]
        if abs(v[0] - t) <= 1e-9 * t:
            return np.concatenate([n.ravel(), [v[4], v[5]]])
    raise SystemExit(f"time {t} not found in {origem}/superficie.dat")


GUARDADOS = ("perfil.dat", "superficie.dat") + EEDF_ARQUIVOS


def guarda_antes_da_retomada(dir_, t):
    """Restart in the same directory: copies of the whole files, deleted only when
    the new run ends successfully, and the previous parametros.json kept as
    parametros_ate_<t>.json."""
    for nome in GUARDADOS:
        if (dir_ / nome).exists():
            shutil.copy2(dir_ / nome, dir_ / (nome + ".antes_da_retomada"))
    if (dir_ / "parametros.json").exists():
        os.replace(dir_ / "parametros.json", dir_ / f"parametros_ate_{t:g}.json")


def escreve_atomico(caminho, texto):
    temporario = caminho.with_name(caminho.name + ".tmp")
    temporario.write_text(texto)
    os.replace(temporario, caminho)


def termina(dir_, inicio):
    """Successful end: deletes the copies kept before the restart."""
    for nome in GUARDADOS:
        (dir_ / (nome + ".antes_da_retomada")).unlink(missing_ok=True)
    print(f"total time: {time.time() - inicio:.1f} s")


def corta_depois_de(dir_, t):
    """Restart in the same directory: keeps the records up to t, inclusive, in
    superficie.dat and in the block files (profiles and EEDF)."""
    linhas = (dir_ / "superficie.dat").read_text().splitlines(keepends=True)
    escreve_atomico(dir_ / "superficie.dat", linhas[0] + "".join(
        l for l in linhas[1:] if float(l.split()[0]) <= t * (1 + 1e-9)))
    for nome in ("perfil.dat",) + EEDF_ARQUIVOS:
        if not (dir_ / nome).exists():
            continue
        blocos, atual = [], []
        for l in (dir_ / nome).read_text().splitlines(keepends=True) + ["\n"]:
            if l.strip():
                atual.append(l)
            elif atual:
                blocos.append(atual)
                atual = []
        escreve_atomico(dir_ / nome, "".join("".join(b) + "\n" for b in blocos
                                            if float(b[0].split()[0]) <= t * (1 + 1e-9)))


def passo_maximo(a):
    """Maximum step of the integrator: with repeated pulses, the shortest part of
    the trapezoid, so that a long step between pulses does not skip the next."""
    if a.forma == "pulso":
        return min(a.subida, a.largura, a.descida)
    return np.inf


def passo_maximo_em(a, t):
    """Maximum step at time t. With --waveform table, inside the window of the
    table it is the smallest spacing of its points; outside, the time left until
    the next window, so that a long step between pulses does not skip a pulse.
    With the other waveforms, the fixed passo_maximo."""
    if a.forma != "tabela":
        return passo_maximo(a)
    t_tab = a.tabela[0]
    fino = float(np.min(np.diff(t_tab)))
    s = t % a.periodo
    if s < t_tab[0]:
        falta = t_tab[0] - s
    elif s <= t_tab[-1]:
        return fino
    else:
        falta = a.periodo - s + t_tab[0]
    # right at the start of the window, the floating point remainder would give a tiny step
    return fino if falta < 1e-3 * fino else falta


def relata(md, t, avaliacoes, jacobianos, fatoracoes, inicio):
    """Progress line at each output: cost of the integrator and evaluations
    outside the table (above and below the mean energy range, and outside the
    T_g range)."""
    tab = md.tab_e if md.energia else md.tab
    abaixo = tab.fora_abaixo if md.energia else "-"
    fora = f"{tab.fora}/{abaixo}/{getattr(tab, 'fora_t', 0)}"
    print(f"t = {t:.4e} s  evaluations {avaliacoes}  jacobians {jacobianos}  factorizations {fatoracoes}  "
          f"out of table (above/below/T_g) {fora}  {time.time() - inicio:.1f} s", flush=True)


def integra_sem_reinicio(a, md, y, t_ini, atol, saidas, inicio):
    """A single BDF run from start to end, with the outputs interpolated by the
    dense output (no internal step stored). With --jac dae the potential is in
    the state and the integrator is BDFMassa, with the Poisson equation in the
    rows of zero mass and outside the error control; with --jac structured,
    scipy's BDF with the Jacobian J_local + S*G."""
    if a.jac == "dae":
        dae = SistemaDAE(md)
        z = dae.estado(t_ini, y)
        massa = np.concatenate([np.ones(dae.N), np.zeros(dae.n_phi)])
        atol_z = np.concatenate([atol, np.full(dae.n_phi, 1.0e30)])
        solver = BDFMassa(dae.rhs, t_ini, z, a.t_end, massa, rtol=a.rtol, atol=atol_z, max_step=passo_maximo(a),
                          jac=lambda t, z: dae.jacobiano(t, z, a.atol))
        so_y = lambda z: z[:dae.N]  # noqa: E731
    else:
        solver = BDF(md.rhs, t_ini, y, a.t_end, rtol=a.rtol, atol=atol, max_step=passo_maximo(a),
                     jac=lambda t, y: md.jacobiano(t, y, a.atol))
        so_y = lambda z: z  # noqa: E731
    k = 1
    while solver.status == "running" and k < len(saidas):
        if a.forma == "tabela":
            solver.max_step = passo_maximo_em(a, solver.t)
        mensagem = solver.step()
        avisa(md, solver.t, so_y(solver.y))
        if solver.status == "failed":
            print(f"failed at t = {solver.t:.4e}: {mensagem}")
            sys.exit(1)
        if solver.t >= saidas[k] * (1 - 1e-12):
            denso = solver.dense_output()
            while k < len(saidas) and solver.t >= saidas[k] * (1 - 1e-12):
                z_saida = denso(saidas[k]) if solver.t > saidas[k] else solver.y
                grava(a.dir, md, [saidas[k]], so_y(z_saida)[:, None], False)
                relata(md, saidas[k], solver.nfev, solver.njev, solver.nlu, inicio)
                k += 1


ARQUIVOS_DO_CASO = ("tabela_energia_tg.dat", "tabela_eedf.dat", "transporte.json", "gas.json")


def quimica_do_caso(caso):
    """Chemistry of the case and the bolsigdb.dat of the thresholds: the
    zdplaskin_m.F90 (with bolsigdb.dat) when it exists; otherwise the
    quimica.json exported by ferramentas/caso.py, which carries the thresholds
    and needs neither file (the distribution cannot carry zdplaskin_m.F90, which
    is ZDPlasKin code)."""
    if (caso / "zdplaskin_m.F90").exists():
        return Quimica(caso / "zdplaskin_m.F90"), caso / "bolsigdb.dat"
    return Quimica.de_json(caso / "quimica.json"), None


def faltam_da_quimica(caso):
    if (caso / "zdplaskin_m.F90").exists():
        return [] if (caso / "bolsigdb.dat").exists() else ["bolsigdb.dat"]
    return [] if (caso / "quimica.json").exists() else ["quimica.json (or zdplaskin_m.F90 and bolsigdb.dat)"]


def confere_composicao(caso):
    """The species with positive fraction in composicao.txt (those the BOLSIG+
    table includes) must be the background of gas.json (those the model leaves
    out of the reactions): otherwise losses are counted twice or not at all.
    Without composicao.txt, the generator uses pure argon. Returns the error
    message or None."""
    arq = caso / "composicao.txt"
    tabela = {}
    if arq.exists():
        for linha in arq.read_text().splitlines():
            partes = linha.split()
            if partes and not partes[0].startswith("#") and len(partes) > 1 and partes[1] != "traco":
                tabela[partes[0].upper()] = float(partes[1])
    else:
        tabela = {"AR": 1.0}
    fundo = {k.upper(): v for k, v in json.loads((caso / "gas.json").read_text()).get("fundo", {}).items()}
    if set(tabela) != set(fundo):
        return (f"{arq.name} (gas of the BOLSIG+ table: {', '.join(sorted(tabela))}) differs from the background of "
                f"gas.json ({', '.join(sorted(fundo))}) in {caso}")
    # in a mixture, the table holds for the fractions it was generated with
    soma = sum(tabela.values())
    for nome, fracao in fundo.items():
        if abs(tabela[nome] / soma - fracao) > 1e-6:
            return (f"{arq.name}: fraction of {nome} in the BOLSIG+ table ({tabela[nome] / soma:g}) differs from "
                    f"that of gas.json ({fracao:g}) in {caso}")
    return None


def valida(p, a):
    """Refuses meaningless parameters, as the input routine of the Fortran code did."""
    regras = [
        (a.n_celulas >= 3, "--cells must be at least 3"),
        (a.gap > 0.0, "--gap must be positive"),
        (a.periodo > 0.0, "--period must be positive"),
        (a.forma in ("seno", "tabela") or a.subida > 0.0, "--rise must be positive with --waveform step or pulse"),
        (a.forma != "tabela" or a.tensao_tabela is not None, "--waveform table needs --voltage-table"),
        (a.forma != "tabela" or a.tensao_tabela is None or a.tensao_tabela.is_file(),
         "--voltage-table: file not found"),
        (a.forma != "tabela" or a.jac != "fd", "--waveform table needs --jac dae or structured"),
        (a.forma != "pulso" or (a.largura > 0.0 and a.descida > 0.0),
         "--width and --fall must be positive with --waveform pulse"),
        (a.forma != "pulso" or a.subida + a.largura + a.descida < a.periodo,
         "--waveform pulse: rise + width + fall must be shorter than --period"),
        (a.t_end > 0.0, "--t-end must be positive"),
        (a.dt_saida > 0.0, "--dt-output must be positive"),
        (a.espessura >= 0.0, "--thickness cannot be negative (zero: bare metal electrode)"),
        (a.espessura_dir is None or a.espessura_dir >= 0.0, "--thickness-right cannot be negative"),
        (a.espessura_dir is None or (a.espessura_dir > 0.0) == (a.espessura > 0.0),
         "--thickness-right: a barrier on one side only is not supported (both or none)"),
        (a.eps_dieletrico_dir is None or a.eps_dieletrico_dir >= 1.0, "--eps-dielectric-right must be at least 1"),
        (a.pressao > 0.0, "--pressure must be positive"),
        (a.eps_dieletrico >= 1.0, "--eps-dielectric must be at least 1"),
        (a.eps_gas >= 1.0, "--eps-gas must be at least 1"),
        (a.gama >= 0.0, "--gamma cannot be negative"),
        (0.0 <= a.reflexao_eletrons < 1.0, "--electron-reflection must be in [0, 1)"),
        (a.rtol > 0.0 and a.atol > 0.0, "--rtol and --atol must be positive"),
        (a.k0_ar_ion is None or a.k0_ar_ion > 0.0, "--k0-ar-ion must be positive"),
        (a.en_aquecimento_ar_ion is None or a.en_aquecimento_ar_ion > 0.0,
         "--en-heating-ar-ion must be positive"),
        (a.k0_ar2_ion is None or a.k0_ar2_ion > 0.0, "--k0-ar2-ion must be positive"),
        (not a.transporte or a.transporte.is_file(), "--transport: file not found"),
        (a.caso.is_dir(), f"--case: {a.caso} is not a folder"),
        (a.t_parede > 0.0, "--t-wall must be positive"),
        (a.t_gas > 0.0, "--t-gas must be positive"),
        (a.agrupamento >= 0.0, "--clustering cannot be negative (zero: uniform mesh)"),
        (a.eps_sec >= 0.0, "--eps-sec cannot be negative"),
        (a.ne0 >= 0.0, "--ne0 cannot be negative"),
        (a.ganho_tres_corpos >= 0.0, "--three-body-gain cannot be negative"),
        (not a.retoma or pathlib.Path(a.retoma[0]).is_dir(), "--restart: origin directory not found"),
        (not a.adaptativa or a.dt_malha > 0.0, "--dt-mesh must be positive with --adaptive"),
    ]
    for ok, mensagem in regras:
        if not ok:
            p.error(mensagem)
    for nome, valor in vars(a).items():
        if isinstance(valor, float) and not np.isfinite(valor):
            p.error(f"--{NOMES.get(nome, nome)} must be a finite number")
    a.tabela = None
    if a.forma == "tabela":
        d = np.loadtxt(a.tensao_tabela, ndmin=2)
        if d.shape[1] != 2:
            p.error("--voltage-table needs two columns, t (s) and V (V)")
        if d.shape[0] < 2 or not np.all(np.isfinite(d)):
            p.error("--voltage-table needs at least two lines of finite numbers")
        a.tabela = (d[:, 0], d[:, 1])
        if np.any(np.diff(a.tabela[0]) <= 0.0) or a.tabela[0][-1] >= a.periodo:
            p.error("--voltage-table: times must increase and be shorter than --period")
        if a.tabela[1][0] != 0.0 or a.tabela[1][-1] != 0.0:
            p.error("--voltage-table: the voltage must start and end at zero")
    try:
        t_ini = float(a.retoma[1]) if a.retoma else 0.0
    except ValueError:
        p.error("--restart needs a directory and a time in seconds")
    if a.t_end <= t_ini:
        p.error("--t-end must be later than the starting time")
    saidas = (a.t_end - t_ini) / a.dt_saida
    if abs(saidas - round(saidas)) > 1e-9 * max(1.0, saidas):
        p.error("--t-end minus the starting time must be a multiple of --dt-output")


def sha256(caminho):
    return hashlib.sha256(pathlib.Path(caminho).read_bytes()).hexdigest()


def grava_parametros(a, md):
    """parametros.json in the run directory: arguments (absolute paths), command
    line, commit, sha256 of the data and of the code, library versions and the
    effective values of the model. The keys of the arguments keep the internal
    (Portuguese) names of the options, so that old and new runs read alike."""
    def git(*args):
        return subprocess.run(["git", "-C", str(RAIZ), *args], capture_output=True, text=True, check=True).stdout
    try:
        commit = git("rev-parse", "HEAD").strip()
        diff = git("diff", "HEAD")
        novos = git("ls-files", "--others", "--exclude-standard", "dbd1d", "scripts")
        alteracoes = {"diff_sha256": hashlib.sha256(diff.encode()).hexdigest() if diff else None,
                      "arquivos_novos": novos.split()}
    except (OSError, subprocess.CalledProcessError):
        commit, alteracoes = "desconhecido", None
    argumentos = {}
    for k, v in vars(a).items():
        if k == "tabela":
            continue  # read from --voltage-table, which is recorded with its sha256
        if isinstance(v, pathlib.Path):
            v = str(v.resolve())
        argumentos[k] = v
    caso = pathlib.Path(a.caso).resolve()
    dados = ["tabela_energia.dat", "tabela_energia_tg.dat", "tabela_bolsig.dat", "tabela_eedf.dat",
             "zdplaskin_m.F90", "quimica.json", "bolsigdb.dat", "transporte.json", "gas.json", "kinet.inp",
             "composicao.txt",
             "calor_rapido.txt", "tabela_calor_tg.dat"]
    codigo = sorted(str(f.relative_to(RAIZ)) for f in (RAIZ / "dbd1d").glob("*.py"))
    hashes = {nome: sha256(caso / nome) for nome in dados if (caso / nome).exists()}
    hashes.update({nome: sha256(RAIZ / nome) for nome in codigo})
    if a.faces:
        hashes["faces"] = sha256(a.faces)
    if a.transporte:
        hashes["transporte"] = sha256(a.transporte)
    if a.tensao_tabela:
        hashes["tensao_tabela"] = sha256(a.tensao_tabela)
    if a.retoma and (pathlib.Path(a.retoma[0]) / "parametros.json").exists():
        hashes["parametros_da_origem"] = sha256(pathlib.Path(a.retoma[0]) / "parametros.json")
    import scipy
    registro = {"argumentos": argumentos, "argv": sys.argv, "cwd": os.getcwd(),
                "versao": VERSAO, "commit": commit, "alteracoes_locais": alteracoes, "sha256": hashes,
                "modelo": {"n_gas": md.n_gas, "t_gas": md.t_gas, "transporte": md.transporte,
                           "gama_sec": md.gama, "eps_sec": md.eps_sec, "eps_gas": md.eps_gas,
                           "reflexao_eletrons": md.reflexao_eletrons,
                           "faces": md.faces.tolist()},
                "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
                **{v: os.environ.get(v) for v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")}}
    (a.dir / "parametros.json").write_text(json.dumps(registro, indent=2, ensure_ascii=False) + "\n")


class _Ajuda(argparse.HelpFormatter):
    """Help that shows only the English name of each option (the first one)."""

    def _format_action_invocation(self, action):
        if not action.option_strings:
            return super()._format_action_invocation(action)
        nome = action.option_strings[0]
        if action.nargs == 0:
            return nome
        return f"{nome} {self._format_args(action, self._get_default_metavar_for_optional(action))}"


FORMAS = {"sine": "seno", "step": "degrau", "pulse": "pulso", "table": "tabela"}
JACOBIANOS = {"dae": "dae", "structured": "estruturado", "fd": "fd"}


def _escolha(nomes, texto):
    """Type of an option with fixed values: the English names, or the internal
    (Portuguese) ones of earlier versions, give the internal value."""
    def converte(valor):
        if valor in nomes:
            return nomes[valor]
        if valor in nomes.values():
            return valor
        raise argparse.ArgumentTypeError(f"choose {texto}")
    return converte


def cria_parser():
    """Command line options. Each option has an English name and, as an alias,
    the Portuguese name of earlier versions, which is also the internal name
    (the key in parametros.json). Validation is done by valida."""
    p = argparse.ArgumentParser(formatter_class=_Ajuda,
                                description="One-dimensional fluid model of a dielectric barrier discharge.")

    def op(ingles, portugues, *outros, **kw):
        # outros: earlier English names, accepted as aliases like the Portuguese one
        nomes = ([ingles] if portugues is None or portugues == ingles else [ingles, portugues]) + list(outros)
        kw.setdefault("dest", (portugues or ingles)[2:].replace("-", "_"))
        if kw.get("action") != "store_true":
            kw.setdefault("metavar", ingles[2:].upper().replace("-", "_"))
        p.add_argument(*nomes, **kw)

    p.add_argument("dir", type=pathlib.Path, help="output directory")
    op("--cells", "--n-celulas", type=int, default=100, help="number of cells")
    op("--clustering", "--agrupamento", type=float, default=0.0,
       help="clustering of the cells towards the walls (tanh mapping; 0: uniform mesh)")
    op("--v-peak", "--v-pico", type=float, default=1.0e3, help="amplitude of the applied voltage, V")
    op("--period", "--periodo", type=float, default=1.0e-7, help="period of the sine or of the pulse train, s")
    op("--waveform", "--forma", type=_escolha(FORMAS, "sine, step, pulse or table"), default="seno",
       metavar="{sine,step,pulse,table}",
       help="applied voltage: sine of amplitude --v-peak; step, linear rise in --rise and constant afterwards; "
            "pulse, trapezoid of --rise, plateau --width and --fall, repeated every --period; table, the pulse "
            "of --voltage-table repeated every --period")
    op("--voltage-table", "--tensao-tabela", type=pathlib.Path,
       help="with --waveform table: file with two columns, t (s, within one period) and V (V), starting and "
            "ending at V = 0; --v-peak does not apply")
    op("--t-gas", None, type=float, default=300.0,
       help="initial gas temperature, K; with --pressure it gives the density (with --heating, set --t-wall to "
            "the same value for a preheated gas)")
    op("--rise", "--subida", type=float, default=0.0, help="rise time of the step or of the pulse, s")
    op("--width", "--largura", type=float, default=0.0, help="duration of the plateau of the pulse, s")
    op("--fall", "--descida", type=float, default=0.0, help="fall time of the pulse, s")
    op("--thickness", "--espessura", type=float, default=0.0,
       help="thickness of the dielectric barrier on each electrode, m (0: bare metal electrodes)")
    op("--thickness-right", "--espessura-dir", type=float, default=None,
       help="thickness of the barrier on the right electrode, m (default: --thickness)")
    op("--eps-dielectric-right", "--eps-dieletrico-dir", type=float, default=None,
       help="relative permittivity of the right barrier (default: --eps-dielectric)")
    op("--pressure", "--pressao", type=float, default=101325.0,
       help="gas pressure, Pa; with --t-gas it gives the density N = p/(k T)")
    op("--gap", None, type=float, default=8.0e-4, help="gas gap width, between the barrier surfaces, m")
    op("--eps-dielectric", "--eps-dieletrico", type=float, default=4.8, help="relative permittivity of the barriers")
    op("--eps-gas", None, type=float, default=1.0, help="relative permittivity of the gas")
    op("--gamma", "--gama", type=float, default=0.05, help="secondary electron emission yield per ion")
    op("--electron-reflection", "--reflexao-eletrons", type=float, default=0.0,
       help="reflection coefficient of the electrons at the walls (Hagelaar et al. 2000), in [0, 1)")
    op("--ne0", None, type=float, default=1.0, help="initial electron density, cm-3")
    op("--neutral", "--neutro", action="store_true",
       help="start quasi-neutral: the initial ion of gas.json at the electron density")
    op("--no-wall-boltzmann", "--sem-freio", "--no-wall-brake", action="store_true",
       help="thermal wall flux without the Boltzmann factor over half a cell")
    op("--t-end", None, type=float, default=1.0e-6, help="final time, s")
    op("--dt-output", "--dt-saida", type=float, default=1.0e-9, help="output interval, s")
    op("--rtol", None, type=float, default=1.0e-4, help="relative tolerance of the integrator")
    op("--atol", None, type=float, default=1.0e-10,
       help="absolute tolerance of the densities, cm-3 (also used for the energy density and the temperature); "
            "values above ~1e-4 leave the electrons between avalanches without error control")
    op("--restart", "--retoma", nargs=2, metavar=("ORIGIN", "T"),
       help="start from the state saved in ORIGIN at time T (s), with the same options")
    op("--energy", "--energia", action="store_true",
       help="solve the electron energy equation (tabela_energia_tg.dat, versus mean energy and T_g)")
    op("--eps-sec", None, type=float, default=5.0, help="energy of the secondary electrons, eV")
    op("--free-gas", "--gas-solto", action="store_true",
       help="solve the background gas density, with diffusion and return of atoms at the walls")
    op("--case", "--caso", type=pathlib.Path, default=RAIZ,
       help="case folder (casos/README.md): chemistry, tables, transporte.json and gas.json; default: the "
            "argon case at the repository root")
    op("--transport", "--transporte", type=pathlib.Path,
       help="transport data per species (default: transporte.json of the case folder)")
    op("--k0-ar-ion", None, type=float,
       help="reduced mobility of Ar+ in Ar (Frost), cm2/(V s); overrides transporte.json")
    op("--en-heating-ar-ion", "--en-aquecimento-ar-ion", type=float,
       help="heating field of Ar+ in Frost's formula, Td; overrides transporte.json")
    op("--k0-ar2-ion", None, type=float, help="reduced mobility of Ar2+, cm2/(V s); overrides transporte.json")
    op("--t-wall", "--t-parede", type=float, default=300.0, help="wall temperature with --heating, K")
    op("--three-body-gain", "--ganho-tres-corpos", type=float, default=0.0,
       help="energy (eV) given to the electron left in three-body recombination; non-negative, physically at "
            "most the ionization energy")
    op("--heating", "--aquecimento", action="store_true",
       help="solve the gas temperature at constant volume (walls at --t-wall)")
    op("--adaptive", "--adaptativa", action="store_true", help="moving mesh by equidistribution")
    op("--dt-mesh", "--dt-malha", type=float, default=2.0e-11, help="interval between mesh adaptations, s")
    op("--adaptive-weight", "--peso-adaptativo", type=float, default=1.0, help="weight of the moving-mesh monitor")
    op("--jac", None, type=_escolha(JACOBIANOS, "dae, structured or fd"), default="dae",
       metavar="{dae,structured,fd}",
       help="dae (default): differential-algebraic system with the potential in the state vector and a structured "
            "sparse Jacobian; structured: structured Jacobian with the potential eliminated (J_local + S G); fd: scipy "
            "finite differences")
    op("--faces", None, type=pathlib.Path, help="file with the m+1 face positions of the mesh, from 0 to --gap, in m")
    return p


# internal (Portuguese) name of each option -> its English name, for the messages
NOMES = {acao.dest: acao.option_strings[0][2:] for acao in cria_parser()._actions if acao.option_strings}


def le_faces(p, a):
    """Faces given by --faces (the number of cells follows from them) or None."""
    if not a.faces:
        return None
    faces = np.atleast_1d(np.loadtxt(a.faces))
    a.n_celulas = len(faces) - 1
    if a.n_celulas < 3 or not np.all(np.isfinite(faces)) or np.any(np.diff(faces) <= 0.0):
        p.error("--faces needs at least 4 finite faces that increase")
    if abs(faces[0]) > 1e-12 * a.gap or abs(faces[-1] - a.gap) > 1e-9 * a.gap:
        p.error(f"--faces must go from 0 to --gap ({a.gap:g} m); the file goes from {faces[0]:g} to {faces[-1]:g} m")
    return faces


def confere_caso(a, caso):
    """Files of the case present and composition of the table equal to the background."""
    faltam = faltam_da_quimica(caso) + [n for n in ARQUIVOS_DO_CASO
                                        if not (caso / n).exists() and not (n == "transporte.json" and a.transporte)]
    if faltam:
        sys.exit(f"error: the case folder {caso} lacks {', '.join(faltam)} (ferramentas/caso.py generates the "
                 "chemistry and the tables; transporte.json and gas.json are data of the mixture)")
    erro = confere_composicao(caso)
    if erro:
        sys.exit(f"error: {erro}")


def monta_modelo(a, caso, faces):
    """Model with the tables of the case: local field from the table in E/N and
    T_g and, with --energy, the electron coefficients versus mean energy and T_g."""
    calor = caso / "tabela_calor_tg.dat"
    try:
        quimica, bolsigdb = quimica_do_caso(caso)
        md = ModeloRapido(
            quimica, TabelaTg(caso / "tabela_energia_tg.dat"),
            n_celulas=a.n_celulas, agrupamento=a.agrupamento, v_pico=a.v_pico,
            periodo=a.periodo, espessura=a.espessura, eps_sec=a.eps_sec, gap=a.gap,
            forma=a.forma, subida=a.subida, largura=a.largura, descida=a.descida,
            tabela_tensao=a.tabela if a.forma == "tabela" else None,
            espessura_dir=a.espessura_dir, eps_dieletrico_dir=a.eps_dieletrico_dir,
            n_gas=a.pressao / (KB * a.t_gas) * 1e-6,
            eps_dieletrico=a.eps_dieletrico, gama_sec=a.gama, freio=not a.sem_freio, faces=faces,
            reflexao_eletrons=a.reflexao_eletrons,
            aquecimento=a.aquecimento, gas_solto=a.gas_solto, t_parede=a.t_parede, t_gas=a.t_gas,
            k0_ar=a.k0_ar_ion, en_aquecimento_ar=a.en_aquecimento_ar_ion, k0_ar2=a.k0_ar2_ion,
            ganho_tres_corpos=a.ganho_tres_corpos, eps_gas=a.eps_gas,
            transporte=a.transporte or caso / "transporte.json",
            tabela_energia=TabelaEnergiaTg(caso / "tabela_energia_tg.dat") if a.energia else None,
            bolsigdb=bolsigdb, gas=caso / "gas.json",
            tabela_calor=TabelaEnergiaTg(calor) if calor.exists() else None)
    except (ErroTransporte, ErroGas, ErroQuimica, ValueError) as e:
        # ValueError: an electron table that cannot be interpolated (tabela.py)
        sys.exit(f"error: {e}")
    md.caso = caso
    confere_faixa_de_tg(a, md)
    mostra_avisos_da_quimica(md)
    return md


def mostra_avisos_da_quimica(md):
    """Warnings of the reader of the chemistry (integer division done as in Fortran)."""
    for aviso in md.q.avisos:
        print(f"WARNING: {aviso}", flush=True)


def confere_faixa_de_tg(a, md):
    """Initial T_g and, with heating, the wall temperature within the T_g nodes of
    the table: outside them the table holds the edge node and the electron
    coefficients would be those of another temperature."""
    t_min, t_max = md.tab.t_gas[0], md.tab.t_gas[-1]
    temperaturas = [("--t-gas", a.t_gas)] + ([("--t-wall", a.t_parede)] if a.aquecimento else [])
    for opcao, valor in temperaturas:
        if not t_min <= valor <= t_max:
            sys.exit(f"error: {opcao} {valor:g} K outside the T_g nodes of the table of the case, from {t_min:g} to "
                     f"{t_max:g} K; put the nodes in {md.caso / 't_nos.txt'} (one per line) and rebuild the tables "
                     "with python3 ferramentas/caso.py tables")


def estado_da_retomada(p, a, md):
    """Starting time and state of a restart. Everything that can refuse it comes
    before any file is touched: an origin with heating or free gas requires the
    same here, and the voltage and initial gas stay those of the origin."""
    origem = pathlib.Path(a.retoma[0])
    if (origem / "parametros.json").exists():
        antes = json.loads((origem / "parametros.json").read_text())["argumentos"]
        for opcao in ("aquecimento", "gas_solto", "energia"):
            if antes.get(opcao) and not getattr(a, opcao):
                p.error(f"--{NOMES[opcao]}: the origin used it; restarting without it would drop variables")
        def valor(opcao):
            agora = getattr(a, opcao)
            if isinstance(agora, pathlib.Path):
                agora = str(agora.resolve())  # the origin records the absolute path
            return agora
        for opcao in ("forma", "subida", "largura", "descida", "t_gas", "tensao_tabela"):
            if opcao in antes and antes[opcao] != valor(opcao):
                p.error(f"--{NOMES[opcao]}: the origin used {antes[opcao]}; the restart needs the same value")
        # other options may change (sensitivity from a saved state), but they are reported
        mudou = [f"--{NOMES.get(k, k.replace('_', '-'))} {antes[k]} -> {valor(k)}" for k in sorted(antes)
                 if k not in ("dir", "retoma", "t_end", "dt_saida") and hasattr(a, k) and antes[k] != valor(k)]
        if mudou:
            print("NOTE: the restart changes options of the origin: " + "; ".join(mudou))
    t_ini = float(a.retoma[1])
    return t_ini, estado_gravado(md, origem, t_ini)


def integra_com_reinicio(a, md, y, t_ini, saidas, atol, inicio):
    """--jac fd: one solve_ivp per output interval (with the moving mesh, per
    stretch of dt_malha, with adaptation at the end of each)."""
    t0 = t_ini
    avaliacoes = jacobianos = fatoracoes = 0
    if a.adaptativa:
        y = readapta(md, y, t0, a.peso_adaptativo)
    for t1 in saidas[1:]:
        trechos = max(1, int(np.ceil((t1 - t0) / a.dt_malha - 1e-9))) if a.adaptativa else 1
        marcos = np.linspace(t0, t1, trechos + 1)
        for ta, tb in zip(marcos[:-1], marcos[1:]):
            # t_eval only at the end of the stretch: solve_ivp does not keep the
            # internal steps (with 1200 cells they took several GB); the steps do not change
            sol = solve_ivp(md.rhs, (ta, tb), y, method="BDF", rtol=a.rtol, atol=atol, t_eval=[tb],
                            max_step=passo_maximo(a))
            if not sol.success:
                print(f"failed at t = {ta:.4e}: {sol.message}")
                sys.exit(1)
            avaliacoes += sol.nfev
            jacobianos += sol.njev
            fatoracoes += sol.nlu
            y = sol.y[:, -1].copy()
            # the solver sits in a reference cycle with the Jacobian and the
            # factorization (hundreds of MB with 600 cells); the Python collector
            # frees it only now and then, and memory grew by ~170 MB per interval
            del sol
            gc.collect()
            if a.adaptativa and tb < t1:
                y = readapta(md, y, tb, a.peso_adaptativo)
        t0 = t1
        grava(a.dir, md, [t1], y[:, None], False)
        if a.adaptativa:
            y = readapta(md, y, t1, a.peso_adaptativo)
        relata(md, t1, avaliacoes, jacobianos, fatoracoes, inicio)


def main():
    """Builds the model of the case, starts from the initial state or from a
    restart and integrates up to --t-end, writing every --dt-output."""
    p = cria_parser()
    a = p.parse_args()
    valida(p, a)
    if a.adaptativa and a.jac != "fd":
        sys.exit("the moving mesh exists only with --jac fd")
    if a.adaptativa and a.aquecimento:
        sys.exit("the moving mesh does not transfer the gas temperature")
    a.dir.mkdir(parents=True, exist_ok=True)
    faces = le_faces(p, a)
    caso = a.caso.resolve()
    confere_caso(a, caso)
    md = monta_modelo(a, caso, faces)
    y = md.estado_inicial(a.ne0, a.neutro)
    t_ini = 0.0
    mesmo_dir = bool(a.retoma) and pathlib.Path(a.retoma[0]).resolve() == a.dir.resolve()
    if a.retoma:
        t_ini, y = estado_da_retomada(p, a, md)
    if mesmo_dir:
        guarda_antes_da_retomada(a.dir, t_ini)
        corta_depois_de(a.dir, t_ini)
    grava_parametros(a, md)
    # the energy (eV cm-3) uses the absolute tolerance of the densities times 1 eV
    atol = np.concatenate([np.full(md.n_var * md.m, a.atol), [1.0e-12, 1.0e-12]])
    saidas = t_ini + np.arange(0.0, (a.t_end - t_ini) * (1 + 1e-9), a.dt_saida)
    if not mesmo_dir:
        grava(a.dir, md, saidas[:1], y[:, None], True)
    inicio = time.time()
    if a.jac == "fd":
        integra_com_reinicio(a, md, y, t_ini, saidas, atol, inicio)
    else:
        integra_sem_reinicio(a, md, y, t_ini, atol, saidas, inicio)
    termina(a.dir, inicio)


if __name__ == "__main__":
    main()
