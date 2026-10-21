# SPDX-License-Identifier: GPL-3.0-or-later
"""Roda o modelo acoplado no tempo com o BDF do scipy (passo e ordem adaptativos)
e grava superficie.dat e perfil.dat no formato do código Fortran.

Uso: python3 prototipo/simula.py DIR --v-pico 1e3 --t-end 2e-8 [opções]
     --retoma ORIGEM T  parte do estado gravado em ORIGEM no instante T (s)
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

# uma thread de BLAS por processo: com várias, processos simultâneos disputam
# os núcleos e a fatoração LU do BDF fica centenas de vezes mais lenta
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_var] = "1"

import numpy as np
from scipy.integrate import BDF, solve_ivp

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from bdf_dae import BDFMassa                 # noqa: E402
from jacobiano import ModeloRapido, SistemaDAE  # noqa: E402
from modelo import KB, ErroGas, ErroTransporte  # noqa: E402
from quimica_zdp import ErroQuimica             # noqa: E402
from malha_adaptativa import faces_por_equidistribuicao, monitor, transfere, transfere_energia  # noqa: E402
from quimica_zdp import Quimica              # noqa: E402
from tabela import TabelaEEDF, TabelaEnergiaTg, TabelaTg     # noqa: E402

RAIZ = pathlib.Path(__file__).resolve().parent.parent


def linha(valores):
    # sempre um espaço entre os números: expoentes de três dígitos (E-101)
    # ocupam a largura toda e grudariam no vizinho
    return "".join(f" {v:24.16E}" for v in valores) + "\n"


def cabecalho(nomes):
    return "".join(f"{n:>24s}" for n in nomes) + "\n"


def nomes_do_perfil(md):
    """Colunas do perfil.dat: as do saida.f90 com os mesmos nomes (sem a CFL,
    que não se aplica ao integrador implícito), todas as espécies na ordem do
    ZDPlasKin e, com a equação de energia, n_eps (eV cm-3), que torna a
    retomada exata."""
    especies = [e.upper() for e in md.q.especies]
    return (["Pos", "Temperature", "volt", "De", "Me", "veloc", "diff T", "rho cv", "alfa", "freq", "Te"]
            + especies + ["E/N", "campo"] + (["n_eps"] if md.energia else []))


def avisa(md, t, y):
    """Avisos únicos de grau de ionização acima de 1e-3 (gás preso) e de gás
    acima de 1000 K (com aquecimento), com a posição."""
    for chave, celula, texto in (("ionizacao", md.celula_muito_ionizada(y), "grau de ionizacao acima de 1e-3"),
                                 ("quente", md.celula_quente(y), f"gas acima de {md.T_ALTA:.0f} K")):
        if celula is not None and not getattr(md, "avisou_" + chave, False):
            # nó na numeração do Fortran (parede = 1, primeira célula = 2)
            print(f"AVISO: {texto} em t = {t:.4e} s, no {celula + 2} (x = {md.xc[celula] * 1e6:.2f} um)", flush=True)
            setattr(md, "avisou_" + chave, True)


# EEDF gravada nos nós do Fortran (2, n/2, n-1, com n = células + 2): primeira
# célula, célula floor(m/2) contada a partir de 1 e última célula
EEDF_ARQUIVOS = ("eedf_borda0.dat", "eedf_centro.dat", "eedf_borda1.dat")


def celulas_da_eedf(md):
    return (0, (md.m + 2) // 2 - 2, md.m - 1)


def energia_da_eedf(md, e_n, extra):
    """Energia média da EEDF gravada: a do estado (com --energia) ou a de campo
    local no E/N e na T_g da célula."""
    if md.energia:
        return extra["eps"]
    if not hasattr(md, "tab_e_lfa"):
        md.tab_e_lfa = TabelaEnergiaTg(md.caso / "tabela_energia_tg.dat")
    return md.tab_e_lfa.energia_de_campo_local(e_n, extra["tg"])


def grava_eedf(dir_, md, t, primeiro, e_n, extra):
    """EEDF da tabela nas três células, pela energia média (com --energia) ou
    pela de campo local (docs/saidas.md: o Fortran a dá do E/N local)."""
    if not hasattr(md, "tab_eedf"):
        md.tab_eedf = TabelaEEDF(md.caso / "tabela_eedf.dat")
    eps = energia_da_eedf(md, e_n, extra)
    for nome, c in zip(EEDF_ARQUIVOS, celulas_da_eedf(md)):
        u, f = md.tab_eedf(eps[c])
        with open(dir_ / nome, "w" if primeiro else "a") as arq:
            arq.write(f"{t:24.16E}\n" + cabecalho(["Energy", "EEDF"]))
            arq.write("".join(linha([a, b]) for a, b in zip(u, f)) + "\n")


def grava(dir_, md, ts, ys, primeiro):
    """Acrescenta (ou, no primeiro registro, cria) superficie.dat, perfil.dat
    e as EEDF com os estados ys (colunas) nos instantes ts, no formato do
    saida.f90 do Fortran."""
    modo = "w" if primeiro else "a"
    with open(dir_ / "superficie.dat", modo) as sup, open(dir_ / "perfil.dat", modo) as per:
        if primeiro:
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
                    print(f"AVISO: energia media fora da tabela em t = {t:.4e} s: {acima} celulas acima, "
                          f"{abaixo} abaixo", flush=True)
            avisa(md, t, y)
            grava_eedf(dir_, md, t, primeiro and t == ts[0], e_n, extra)
            kap = md.condutividade(t_gas)
            rho_cv = md.cv_por_kb * KB * md.n_pesadas(n) * 1e6
            # densidades de todas as espécies; o gás preso com a sua densidade
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
                    # nós das paredes, como o Fortran: temperatura e kappa da parede,
                    # rho cv da célula vizinha, coeficientes e densidades nulos
                    viz = 0 if i == 0 else md.m - 1
                    valores = [x[i], t_par, phi[i], 0.0, 0.0, 0.0, k_par, rho_cv[viz],
                               k_par / rho_cv[viz], 0.0, 0.0, *np.zeros(len(md.q.especies)), 0.0,
                               campo[min(i, md.m)]]
                    if md.energia:
                        valores.append(0.0)
                per.write(linha(valores))
            per.write("\n")


def readapta(md, y, t, peso):
    """Malha nova por equidistribuição (ln das densidades carregadas, com piso de
    1e-6 do máximo e nunca abaixo de 1e8 cm-3, densidades que não pesam no
    campo; e campo nas células) e estado transferido de forma conservativa para
    ela; a energia dos elétrons vai pela energia média ponderada. O modelo passa
    a usar a malha nova."""
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
    """Estado y no instante t a partir de perfil.dat e superficie.dat. Recusa
    origem com outra malha; a energia vem da coluna n_eps (exata) ou, em
    arquivos antigos, da Te gravada."""
    sys.path.insert(0, str(RAIZ / "scripts"))
    from resumo import blocos
    for tb, nomes, linhas in blocos(origem / "perfil.dat"):
        if abs(tb - t) <= 1e-9 * t:
            interior = np.array(linhas[1:-1])
            if len(interior) != md.m or not np.allclose(interior[:, 0], md.xc, rtol=1e-12, atol=0):
                raise SystemExit(f"a malha de {origem} difere da desta execucao")
            faltam = [c for c in md.nomes if c not in nomes]
            if md.aquecimento and "Temperature" not in nomes and "Tgas" not in nomes:
                faltam.append("Temperature")
            if faltam:
                raise SystemExit(f"{origem}/perfil.dat nao tem as colunas {', '.join(faltam)}")
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
        raise SystemExit(f"instante {t} nao encontrado em {origem}/perfil.dat")
    for l in (origem / "superficie.dat").read_text().splitlines()[1:]:
        v = [float(x) for x in l.split()]
        if abs(v[0] - t) <= 1e-9 * t:
            return np.concatenate([n.ravel(), [v[4], v[5]]])
    raise SystemExit(f"instante {t} nao encontrado em {origem}/superficie.dat")


GUARDADOS = ("perfil.dat", "superficie.dat") + EEDF_ARQUIVOS


def guarda_antes_da_retomada(dir_, t):
    """Retomada no próprio diretório: cópias dos arquivos inteiros, apagadas só
    quando a nova execução termina bem, e o parametros.json anterior guardado
    como parametros_ate_<t>.json."""
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
    """Fim bem-sucedido: apaga as cópias guardadas antes da retomada."""
    for nome in GUARDADOS:
        (dir_ / (nome + ".antes_da_retomada")).unlink(missing_ok=True)
    print(f"tempo total: {time.time() - inicio:.1f} s")


def corta_depois_de(dir_, t):
    """Retomada no próprio diretório: mantém os registros até t, inclusive, no
    superficie.dat e nos arquivos em blocos (perfil e EEDF)."""
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
    """Passo máximo do integrador: com pulsos repetidos, o menor trecho do
    trapézio, para o passo largo entre pulsos não pular o pulso seguinte."""
    if a.forma == "pulso":
        return min(a.subida, a.largura, a.descida)
    return np.inf


def passo_maximo_em(a, t):
    """Passo máximo no instante t. Com --forma tabela, dentro da janela da
    tabela a menor separação entre os seus pontos; fora dela, a distância até
    o início da janela seguinte, para o passo largo entre pulsos não pular o
    pulso. Nas outras formas, o passo_maximo fixo."""
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
    # rente ao início da janela, o resto da divisão em ponto flutuante daria um passo ínfimo
    return fino if falta < 1e-3 * fino else falta


def relata(md, t, avaliacoes, jacobianos, fatoracoes, inicio):
    """Linha de andamento a cada saída: custo do integrador e avaliações fora
    da tabela (acima e abaixo da energia média, e fora da faixa de T_g)."""
    tab = md.tab_e if md.energia else md.tab
    abaixo = tab.fora_abaixo if md.energia else "-"
    fora = f"{tab.fora}/{abaixo}/{getattr(tab, 'fora_t', 0)}"
    print(f"t = {t:.4e} s  avaliacoes {avaliacoes}  jacobianos {jacobianos}  fatoracoes {fatoracoes}  "
          f"avaliacoes fora da tabela (acima/abaixo/T_g) {fora}  {time.time() - inicio:.1f} s", flush=True)


def integra_sem_reinicio(a, md, y, t_ini, atol, saidas, inicio):
    """Um único BDF do início ao fim, com as saídas interpoladas pela saída densa
    (nenhum passo interno guardado). Com --jac dae o potencial entra no estado e
    o integrador é o BDFMassa, com o Poisson nas linhas de massa nula e fora do
    controle de erro; com --jac estruturado, o BDF do scipy com o jacobiano
    J_local + S*G."""
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
            print(f"falhou em t = {solver.t:.4e}: {mensagem}")
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
    """Química do caso e o bolsigdb.dat dos limiares: o zdplaskin_m.F90 (com o
    bolsigdb.dat) quando existe, que é a fonte no repositório de pesquisa; senão
    o quimica.json exportado por ferramentas/caso.py, que leva os limiares e
    dispensa os dois arquivos (o pacote distribuído não pode levar o
    zdplaskin_m.F90, que é código do ZDPlasKin)."""
    if (caso / "zdplaskin_m.F90").exists():
        return Quimica(caso / "zdplaskin_m.F90"), caso / "bolsigdb.dat"
    return Quimica.de_json(caso / "quimica.json"), None


def faltam_da_quimica(caso):
    if (caso / "zdplaskin_m.F90").exists():
        return [] if (caso / "bolsigdb.dat").exists() else ["bolsigdb.dat"]
    return [] if (caso / "quimica.json").exists() else ["quimica.json (ou zdplaskin_m.F90 e bolsigdb.dat)"]


def confere_composicao(caso):
    """As espécies com fração positiva na composicao.txt (as que a tabela do
    BOLSIG+ inclui) precisam ser as do fundo do gas.json (as que o modelo deixa
    de fora das reações): senão perdas contadas duas vezes ou nenhuma. Sem
    composicao.txt, o gerador usa argônio puro. Devolve a mensagem de erro ou None."""
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
        return (f"{arq.name} (gás da tabela do BOLSIG+: {', '.join(sorted(tabela))}) não coincide com o fundo do "
                f"gas.json ({', '.join(sorted(fundo))}) em {caso}")
    # numa mistura, a tabela vale para as frações com que foi gerada
    soma = sum(tabela.values())
    for nome, fracao in fundo.items():
        if abs(tabela[nome] / soma - fracao) > 1e-6:
            return (f"{arq.name}: fração de {nome} na tabela do BOLSIG+ ({tabela[nome] / soma:g}) diferente da do "
                    f"gas.json ({fracao:g}) em {caso}")
    return None


def valida(p, a):
    """Recusa parâmetros sem sentido, como le_entrada no Fortran (entrada.f90)."""
    regras = [
        (a.n_celulas >= 3, "--n-celulas precisa ser pelo menos 3"),
        (a.gap > 0.0, "--gap precisa ser positivo"),
        (a.periodo > 0.0, "--periodo precisa ser positivo"),
        (a.forma in ("seno", "tabela") or a.subida > 0.0, "--subida precisa ser positiva com --forma degrau ou pulso"),
        (a.forma != "tabela" or a.tensao_tabela is not None, "--forma tabela precisa de --tensao-tabela"),
        (a.forma != "tabela" or a.tensao_tabela is None or a.tensao_tabela.is_file(),
         "--tensao-tabela: arquivo não encontrado"),
        (a.forma != "tabela" or a.jac != "fd", "--forma tabela precisa de --jac dae ou estruturado"),
        (a.forma != "pulso" or (a.largura > 0.0 and a.descida > 0.0),
         "--largura e --descida precisam ser positivas com --forma pulso"),
        (a.forma != "pulso" or a.subida + a.largura + a.descida < a.periodo,
         "--forma pulso: subida + largura + descida precisa ser menor que o --periodo"),
        (a.t_end > 0.0, "--t-end precisa ser positivo"),
        (a.dt_saida > 0.0, "--dt-saida precisa ser positivo"),
        (a.espessura >= 0.0, "--espessura nao pode ser negativa (zero: eletrodo metalico)"),
        (a.espessura_dir is None or a.espessura_dir >= 0.0, "--espessura-dir nao pode ser negativa"),
        (a.espessura_dir is None or (a.espessura_dir > 0.0) == (a.espessura > 0.0),
         "--espessura-dir: barreira so de um lado nao e suportada (as duas ou nenhuma)"),
        (a.eps_dieletrico_dir is None or a.eps_dieletrico_dir >= 1.0, "--eps-dieletrico-dir precisa ser pelo menos 1"),
        (a.pressao > 0.0, "--pressao precisa ser positiva"),
        (a.eps_dieletrico >= 1.0, "--eps-dieletrico precisa ser pelo menos 1"),
        (a.eps_gas >= 1.0, "--eps-gas precisa ser pelo menos 1"),
        (a.gama >= 0.0, "--gama nao pode ser negativo"),
        (a.rtol > 0.0 and a.atol > 0.0, "--rtol e --atol precisam ser positivos"),
        (a.k0_ar_ion is None or a.k0_ar_ion > 0.0, "--k0-ar-ion precisa ser positivo"),
        (a.en_aquecimento_ar_ion is None or a.en_aquecimento_ar_ion > 0.0,
         "--en-aquecimento-ar-ion precisa ser positivo"),
        (a.k0_ar2_ion is None or a.k0_ar2_ion > 0.0, "--k0-ar2-ion precisa ser positivo"),
        (not a.transporte or a.transporte.is_file(), "--transporte: arquivo inexistente"),
        (a.caso.is_dir(), f"--caso: {a.caso} não é uma pasta"),
        (a.t_parede > 0.0, "--t-parede precisa ser positiva"),
        (a.t_gas > 0.0, "--t-gas precisa ser positiva"),
        (a.agrupamento >= 0.0, "--agrupamento nao pode ser negativo (zero: malha uniforme)"),
        (a.eps_sec >= 0.0, "--eps-sec nao pode ser negativa"),
        (a.ne0 >= 0.0, "--ne0 nao pode ser negativa"),
        (a.ganho_tres_corpos >= 0.0, "--ganho-tres-corpos nao pode ser negativo"),
        (not a.retoma or pathlib.Path(a.retoma[0]).is_dir(), "--retoma: diretorio de origem inexistente"),
        (not a.adaptativa or a.dt_malha > 0.0, "--dt-malha precisa ser positivo com --adaptativa"),
    ]
    for ok, mensagem in regras:
        if not ok:
            p.error(mensagem)
    a.tabela = None
    if a.forma == "tabela":
        d = np.loadtxt(a.tensao_tabela, ndmin=2)
        if d.shape[1] != 2:
            p.error("--tensao-tabela precisa de duas colunas, t (s) e V (V)")
        a.tabela = (d[:, 0], d[:, 1])
        if np.any(np.diff(a.tabela[0]) <= 0.0) or a.tabela[0][-1] >= a.periodo:
            p.error("--tensao-tabela: instantes crescentes e menores que o --periodo")
        if a.tabela[1][0] != 0.0 or a.tabela[1][-1] != 0.0:
            p.error("--tensao-tabela: a tensão precisa começar e terminar em zero")
    try:
        t_ini = float(a.retoma[1]) if a.retoma else 0.0
    except ValueError:
        p.error("--retoma precisa de um diretorio e de um instante em segundos")
    if a.t_end <= t_ini:
        p.error("--t-end precisa ser maior que o instante de partida")
    saidas = (a.t_end - t_ini) / a.dt_saida
    if abs(saidas - round(saidas)) > 1e-9 * max(1.0, saidas):
        p.error("--t-end menos o instante de partida precisa ser multiplo de --dt-saida")


def sha256(caminho):
    return hashlib.sha256(pathlib.Path(caminho).read_bytes()).hexdigest()


def grava_parametros(a, md):
    """parametros.json no diretório da execução: argumentos (caminhos
    absolutos), linha de comando, commit, sha256 dos dados e do código, versões
    e os valores efetivos do modelo."""
    def git(*args):
        return subprocess.run(["git", "-C", str(RAIZ), *args], capture_output=True, text=True, check=True).stdout
    try:
        commit = git("rev-parse", "HEAD").strip()
        diff = git("diff", "HEAD")
        novos = git("ls-files", "--others", "--exclude-standard", "prototipo", "scripts")
        alteracoes = {"diff_sha256": hashlib.sha256(diff.encode()).hexdigest() if diff else None,
                      "arquivos_novos": novos.split()}
    except (OSError, subprocess.CalledProcessError):
        commit, alteracoes = "desconhecido", None
    argumentos = {}
    for k, v in vars(a).items():
        if k == "tabela":
            continue  # lida do --tensao-tabela, que fica registrado com o sha256
        if isinstance(v, pathlib.Path):
            v = str(v.resolve())
        argumentos[k] = v
    caso = pathlib.Path(a.caso).resolve()
    dados = ["tabela_energia.dat", "tabela_energia_tg.dat", "tabela_bolsig.dat", "tabela_eedf.dat",
             "zdplaskin_m.F90", "quimica.json", "bolsigdb.dat", "transporte.json", "gas.json", "kinet.inp",
             "composicao.txt",
             "calor_rapido.txt", "tabela_calor_tg.dat"]
    codigo = sorted(str(f.relative_to(RAIZ)) for f in (RAIZ / "prototipo").glob("*.py"))
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
                "commit": commit, "alteracoes_locais": alteracoes, "sha256": hashes,
                "modelo": {"n_gas": md.n_gas, "t_gas": md.t_gas, "transporte": md.transporte,
                           "gama_sec": md.gama, "eps_sec": md.eps_sec, "eps_gas": md.eps_gas,
                           "faces": md.faces.tolist()},
                "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
                **{v: os.environ.get(v) for v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")}}
    (a.dir / "parametros.json").write_text(json.dumps(registro, indent=2, ensure_ascii=False) + "\n")


def cria_parser():
    """Opções de linha de comando; a validação é a de valida."""
    p = argparse.ArgumentParser()
    p.add_argument("dir", type=pathlib.Path)
    p.add_argument("--n-celulas", type=int, default=100)
    p.add_argument("--agrupamento", type=float, default=0.0)
    p.add_argument("--v-pico", type=float, default=1.0e3)
    p.add_argument("--periodo", type=float, default=1.0e-7, help="período da senoide ou da repetição dos pulsos, s")
    p.add_argument("--forma", choices=("seno", "degrau", "pulso", "tabela"), default="seno",
                   help="tensão aplicada: senoide de amplitude --v-pico; degrau, subida linear em --subida e "
                        "constante depois; pulso, trapézio de --subida, platô --largura e --descida, repetido a "
                        "cada --periodo; tabela, o pulso de --tensao-tabela repetido a cada --periodo")
    p.add_argument("--tensao-tabela", type=pathlib.Path,
                   help="com --forma tabela: arquivo de duas colunas, t (s, dentro de um período) e V (V), "
                        "começando e terminando em V = 0; --v-pico não se aplica")
    p.add_argument("--t-gas", type=float, default=300.0,
                   help="temperatura inicial do gás, K; a densidade sai dela a 1 atm (com --aquecimento, "
                        "use --t-parede igual para gás pré-aquecido)")
    p.add_argument("--subida", type=float, default=0.0, help="tempo de subida do degrau ou do pulso, s")
    p.add_argument("--largura", type=float, default=0.0, help="duração do platô do pulso, s")
    p.add_argument("--descida", type=float, default=0.0, help="tempo de descida do pulso, s")
    p.add_argument("--espessura", type=float, default=0.0, help="barreira dielétrica de cada eletrodo, m")
    p.add_argument("--espessura-dir", type=float, default=None,
                   help="barreira do eletrodo direito, m (padrão: a de --espessura)")
    p.add_argument("--eps-dieletrico-dir", type=float, default=None,
                   help="permissividade relativa da barreira direita (padrão: a de --eps-dieletrico)")
    p.add_argument("--pressao", type=float, default=101325.0,
                   help="pressão do gás, Pa; com --t-gas dá a densidade N = p/(k T)")
    p.add_argument("--gap", type=float, default=8.0e-4, help="distância entre as superfícies, m")
    p.add_argument("--eps-dieletrico", type=float, default=4.8)
    p.add_argument("--eps-gas", type=float, default=1.0,
                   help="permissividade relativa do gás (o Fortran usa 1,000516; docs/lacunas.md, T4)")
    p.add_argument("--gama", type=float, default=0.05, help="emissão secundária por íon")
    p.add_argument("--ne0", type=float, default=1.0, help="densidade inicial de elétrons, cm-3")
    p.add_argument("--neutro", action="store_true", help="íon inicial do gas.json igual aos elétrons")
    p.add_argument("--sem-freio", action="store_true", help="fluxo térmico de parede sem o fator de Boltzmann")
    p.add_argument("--t-end", type=float, default=1.0e-6, help="instante final, s")
    p.add_argument("--dt-saida", type=float, default=1.0e-9)
    p.add_argument("--rtol", type=float, default=1.0e-4, help="tolerância relativa do integrador")
    p.add_argument("--atol", type=float, default=1.0e-10,
                   help="tolerância absoluta das densidades, cm-3 (e da energia e da temperatura); com 1e-4 "
                        "os elétrons esgotados entre avalanches ficavam sem controle de erro e n_e errava até "
                        "um fator 85 (ACHADOS, Fortran contra protótipo)")
    p.add_argument("--retoma", nargs=2, metavar=("ORIGEM", "T"))
    p.add_argument("--energia", action="store_true",
                   help="equação de energia dos elétrons (tabela_energia_tg.dat, por energia média e T_g)")
    p.add_argument("--eps-sec", type=float, default=5.0, help="energia dos elétrons secundários, eV")
    p.add_argument("--gas-solto", action="store_true",
                   help="gás de fundo no estado, com difusão e devolução de átomos pela parede")
    p.add_argument("--caso", type=pathlib.Path, default=RAIZ,
                   help="pasta do caso (casos/LEIAME.md): química, tabelas, bolsigdb.dat, transporte.json e "
                        "gas.json; padrão: a raiz, o argônio")
    p.add_argument("--transporte", type=pathlib.Path,
                   help="dados de transporte por espécie (padrão: transporte.json da pasta do caso)")
    p.add_argument("--k0-ar-ion", type=float,
                   help="mobilidade reduzida do Ar+ em Ar (Frost), cm2/(V s); sobrepõe o transporte.json")
    p.add_argument("--en-aquecimento-ar-ion", type=float,
                   help="campo de aquecimento do Ar+ na fórmula de Frost, Td; sobrepõe o transporte.json")
    p.add_argument("--k0-ar2-ion", type=float,
                   help="mobilidade reduzida do Ar2+, cm2/(V s); sobrepõe o transporte.json")
    p.add_argument("--t-parede", type=float, default=300.0, help="temperatura das paredes com --aquecimento, K")
    p.add_argument("--ganho-tres-corpos", type=float, default=0.0,
                   help="energia (eV) do elétron que sobra na recombinação a três corpos; de 0 à energia de "
                        "ionização do gás")
    p.add_argument("--aquecimento", action="store_true",
                   help="temperatura do gás no estado, a volume constante (paredes a --t-parede)")
    p.add_argument("--adaptativa", action="store_true", help="malha móvel por equidistribuição")
    p.add_argument("--dt-malha", type=float, default=2.0e-11, help="intervalo entre readaptações, s")
    p.add_argument("--peso-adaptativo", type=float, default=1.0)
    p.add_argument("--jac", choices=["dae", "estruturado", "fd"], default="dae",
                   help="dae: potencial no estado, jacobiano esparso (padrão); estruturado: "
                        "jacobiano J_local + S*G; fd: diferenças finitas do scipy (original)")
    p.add_argument("--faces", type=pathlib.Path, help="arquivo com as m+1 faces da malha, m")
    return p


def le_faces(p, a):
    """Faces dadas por --faces (o número de células sai delas) ou None."""
    if not a.faces:
        return None
    faces = np.loadtxt(a.faces)
    a.n_celulas = len(faces) - 1
    if a.n_celulas < 3 or np.any(np.diff(faces) <= 0.0):
        p.error("--faces precisa de pelo menos 4 faces em ordem crescente")
    return faces


def confere_caso(a, caso):
    """Arquivos do caso presentes e composição da tabela igual ao fundo."""
    faltam = faltam_da_quimica(caso) + [n for n in ARQUIVOS_DO_CASO
                                        if not (caso / n).exists() and not (n == "transporte.json" and a.transporte)]
    if faltam:
        sys.exit(f"erro: a pasta do caso {caso} não tem {', '.join(faltam)} (ferramentas/caso.py gera a química "
                 "e as tabelas; transporte.json e gas.json são dados da mistura)")
    erro = confere_composicao(caso)
    if erro:
        sys.exit(f"erro: {erro}")


def monta_modelo(a, caso, faces):
    """Modelo com as tabelas do caso: campo local pela tabela em E/N e T_g (G4)
    e, com --energia, os coeficientes dos elétrons por energia média e T_g."""
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
            aquecimento=a.aquecimento, gas_solto=a.gas_solto, t_parede=a.t_parede, t_gas=a.t_gas,
            k0_ar=a.k0_ar_ion, en_aquecimento_ar=a.en_aquecimento_ar_ion, k0_ar2=a.k0_ar2_ion,
            ganho_tres_corpos=a.ganho_tres_corpos, eps_gas=a.eps_gas,
            transporte=a.transporte or caso / "transporte.json",
            tabela_energia=TabelaEnergiaTg(caso / "tabela_energia_tg.dat") if a.energia else None,
            bolsigdb=bolsigdb, gas=caso / "gas.json",
            tabela_calor=TabelaEnergiaTg(calor) if calor.exists() else None)
    except (ErroTransporte, ErroGas, ErroQuimica) as e:
        sys.exit(f"erro: {e}")
    md.caso = caso
    confere_faixa_de_tg(a, md)
    mostra_avisos_da_quimica(md)
    return md


def mostra_avisos_da_quimica(md):
    """Avisos do leitor do zdplaskin_m.F90 (divisão inteira feita como no Fortran)."""
    for aviso in md.q.avisos:
        print(f"AVISO: {aviso}", flush=True)


def confere_faixa_de_tg(a, md):
    """T_g inicial e, com aquecimento, a das paredes dentro dos nós de T_g da
    tabela: fora deles a tabela prende o nó extremo e os coeficientes dos
    elétrons ficam os de outra temperatura."""
    t_min, t_max = md.tab.t_gas[0], md.tab.t_gas[-1]
    temperaturas = [("--t-gas", a.t_gas)] + ([("--t-parede", a.t_parede)] if a.aquecimento else [])
    for opcao, valor in temperaturas:
        if not t_min <= valor <= t_max:
            sys.exit(f"erro: {opcao} {valor:g} K fora dos nós de T_g da tabela do caso, de {t_min:g} a {t_max:g} K; "
                     f"ponha os nós em {md.caso / 't_nos.txt'} (um por linha) e refaça as tabelas com "
                     "python3 ferramentas/caso.py tabelas")


def estado_da_retomada(p, a, md):
    """Instante e estado de partida da retomada. Tudo o que pode recusá-la vem
    antes de mexer em qualquer arquivo: a origem com aquecimento ou gás solto
    exige o mesmo aqui, e a tensão e o gás inicial continuam os da origem."""
    origem = pathlib.Path(a.retoma[0])
    if (origem / "parametros.json").exists():
        antes = json.loads((origem / "parametros.json").read_text())["argumentos"]
        for opcao in ("aquecimento", "gas_solto"):
            if antes.get(opcao) and not getattr(a, opcao):
                p.error(f"--{opcao.replace('_', '-')}: a origem o usou; retomar sem ele descartaria variaveis")
        for opcao in ("forma", "subida", "largura", "descida", "t_gas", "tensao_tabela"):
            agora = getattr(a, opcao)
            if isinstance(agora, pathlib.Path):
                agora = str(agora.resolve())  # a origem registra o caminho absoluto
            if opcao in antes and antes[opcao] != agora:
                p.error(f"--{opcao.replace('_', '-')}: a origem usou {antes[opcao]}; a retomada precisa do mesmo valor")
    t_ini = float(a.retoma[1])
    return t_ini, estado_gravado(md, origem, t_ini)


def integra_com_reinicio(a, md, y, t_ini, saidas, atol, inicio):
    """--jac fd: um solve_ivp por intervalo de saída (com malha adaptativa, por
    trecho de dt_malha, com readaptação no fim de cada um)."""
    t0 = t_ini
    avaliacoes = jacobianos = fatoracoes = 0
    if a.adaptativa:
        y = readapta(md, y, t0, a.peso_adaptativo)
    for t1 in saidas[1:]:
        trechos = max(1, int(np.ceil((t1 - t0) / a.dt_malha - 1e-9))) if a.adaptativa else 1
        marcos = np.linspace(t0, t1, trechos + 1)
        for ta, tb in zip(marcos[:-1], marcos[1:]):
            # t_eval só com o fim do trecho: o solve_ivp não guarda os passos
            # internos (com 1200 células eles ocupavam vários GB); os passos não mudam
            sol = solve_ivp(md.rhs, (ta, tb), y, method="BDF", rtol=a.rtol, atol=atol, t_eval=[tb],
                            max_step=passo_maximo(a))
            if not sol.success:
                print(f"falhou em t = {ta:.4e}: {sol.message}")
                sys.exit(1)
            avaliacoes += sol.nfev
            jacobianos += sol.njev
            fatoracoes += sol.nlu
            y = sol.y[:, -1].copy()
            # o solucionador fica num ciclo de referências com o jacobiano e a
            # fatoração (centenas de MB com 600 células); o coletor do Python só
            # o libera de tempos em tempos e a memória crescia ~170 MB por intervalo
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
    """Monta o modelo do caso, parte do estado inicial ou da retomada e integra
    até --t-end, gravando a cada --dt-saida."""
    p = cria_parser()
    a = p.parse_args()
    valida(p, a)
    if a.adaptativa and a.jac != "fd":
        sys.exit("a malha adaptativa só existe com --jac fd")
    if a.adaptativa and a.aquecimento:
        sys.exit("a malha adaptativa não transfere a temperatura do gás")
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
    # a energia (eV cm-3) usa a mesma tolerância absoluta das densidades vezes 1 eV
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
