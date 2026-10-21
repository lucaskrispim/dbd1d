# SPDX-License-Identifier: GPL-3.0-or-later
"""Ferramenta de caso: prepara uma mistura gasosa para o protótipo a partir dos
arquivos que o ZDPlasKin já usa, sem editar código para cada gás.

Uma pasta de caso tem o kinet.inp e o bolsigdb.dat da mistura. A ferramenta
gera o zdplaskin_m.F90 pelo pré-processador do ZDPlasKin (versão 2.0a) e aplica as edições
registradas em EDICOES_ZDPLASKIN, as mesmas do zdplaskin_m.F90 do repositório:
tabela interna do BOLSIG+ 10 vezes mais densa e duas mensagens desligadas.

As tabelas dos elétrons (tabela_energia.dat, tabela_eedf.dat,
tabela_energia_tg.dat e a de conferência) saem do BOLSIG+ pelo gerador
ferramentas/gera_tabela_energia.f90, compilado com gfortran (ou FC) contra o
zdplaskin_m.F90 do caso, com a composição do composicao.txt da pasta.

O ZDPlasKin e o BOLSIG+ não vêm com a ferramenta (a licença do ZDPlasKin
permite usar, não redistribuir): ZDPLASKIN_DIR indica a pasta do ZDPlasKin
baixado (pré-processador, dvode_f90_m.F90 e bolsig_x86_64.so) e BOLSIG_SO,
opcional, outra biblioteca do BOLSIG+. Só a geração de um caso novo precisa
deles: caso.py exporta grava o quimica.json, com a química e os limiares, com
o qual o simulador roda sem o zdplaskin_m.F90 e sem o bolsigdb.dat.

Seção de choque com mais de 900 pontos derruba o BOLSIG+: caso.py reduz guarda
o bolsigdb_original.dat e grava o reduzido, com o erro em reducao_secoes.txt;
caso.py tudo reduz sozinho quando precisa. Se o BOLSIG+ cai em E/N alto, a
geração das tabelas grava o en_max.txt abaixo do ponto da queda e repete
(tabelas_topo.txt).

caso.py estima completa o transporte.json com estimativas marcadas na fonte:
mobilidade dos íons pelo limite de Langevin com a lei de Blanc e difusão dos
neutros pela teoria cinética (Cantera); a perda na parede não se estima.

caso.py entalpias completa calor.entalpias do gas.json: estado fundamental
pelo GRI-Mech 3.0 (Cantera) e excitados e íons pelos limiares do BOLSIG+.

Uso: python3 ferramentas/caso.py zdplaskin|calor|tabelas|transporte|estima|entalpias|reduz|exporta|tudo PASTA
"""
import os
import pathlib
import subprocess
import sys
import tempfile

RAIZ = pathlib.Path(__file__).resolve().parent.parent

EDICOES_ZDPLASKIN = [
    ("    bolsig_points_max = int( bolsig_mesh_a * log( bolsig_field_max ) - bolsig_mesh_b )",
     "    bolsig_points_max = int(10*( bolsig_mesh_a * log( bolsig_field_max ) - bolsig_mesh_b) )"),
    ('    if( lprint ) write(*,"(2(A,1pd9.2),A)") "ZDPlasKin INFO: set accuracy", atol_save, " (absolute) &", '
     'rtol_save, " (relative)"',
     '    !if( lprint ) write(*,"(2(A,1pd9.2),A)") "ZDPlasKin INFO: set accuracy", atol_save, " (absolute) &", '
     'rtol_save, " (relative)"'),
    ('  write(*,"(A)") "ZDPlasKin INFO: reset data and configuration"',
     '  !write(*,"(A)") "ZDPlasKin INFO: reset data and configuration"'),
]


class ErroCaso(Exception):
    """Pasta de caso incompleta ou inválida."""


def terceiros():
    """Caminhos do pré-processador e do dvode_f90_m.F90 do ZDPlasKin e da
    biblioteca do BOLSIG+, que não vão no pacote (a licença do ZDPlasKin permite
    usar, não redistribuir): ZDPLASKIN_DIR, a pasta do ZDPlasKin baixado, e
    BOLSIG_SO, a biblioteca (padrão: bolsig_x86_64.so dentro de ZDPLASKIN_DIR).
    Sem as variáveis, as cópias do repositório de pesquisa, se existirem."""
    zdp = os.environ.get("ZDPLASKIN_DIR")
    if zdp:
        base = pathlib.Path(zdp)
        caminhos = {"preprocessor": base / "preprocessor", "dvode": base / "dvode_f90_m.F90",
                    "bolsig": base / "bolsig_x86_64.so"}
    else:
        caminhos = {"preprocessor": RAIZ / "ferramentas" / "zdplaskin" / "preprocessor",
                    "dvode": RAIZ / "dvode_f90_m.F90", "bolsig": RAIZ / "bolsig_x86_64.so"}
    if os.environ.get("BOLSIG_SO"):
        caminhos["bolsig"] = pathlib.Path(os.environ["BOLSIG_SO"])
    faltam = [f"{nome} ({c})" for nome, c in caminhos.items() if not c.exists()]
    if faltam:
        raise ErroCaso("faltam programas de terceiros: " + ", ".join(faltam) + ". Baixe o ZDPlasKin "
                       "(www.zdplaskin.laplace.univ-tlse.fr), que traz o pré-processador, o dvode_f90_m.F90 e "
                       "o BOLSIG+, e indique a pasta em ZDPLASKIN_DIR (e a biblioteca em BOLSIG_SO, se for outra)")
    return caminhos


def gera_zdplaskin(pasta):
    """zdplaskin_m.F90 da pasta a partir do kinet.inp, com as edições registradas."""
    pasta = pathlib.Path(pasta)
    destino = pasta / "zdplaskin_m.F90"
    if not (pasta / "kinet.inp").exists():
        raise ErroCaso(f"{pasta}: falta o kinet.inp")
    destino.unlink(missing_ok=True)
    r = subprocess.run([str(terceiros()["preprocessor"]), "kinet.inp", "zdplaskin_m.F90"], cwd=pasta,
                       stdin=subprocess.DEVNULL, capture_output=True, text=True)
    if r.returncode != 0 or not destino.exists() or "ERROR" in r.stdout + r.stderr:
        raise ErroCaso(f"o pré-processador do ZDPlasKin recusou {pasta / 'kinet.inp'}:\n"
                       + (r.stdout + r.stderr)[-2000:])
    texto = destino.read_text()
    for velho, novo in EDICOES_ZDPLASKIN:
        n = texto.count(velho)
        if n != 1:
            destino.unlink()
            raise ErroCaso(f"edição do zdplaskin_m.F90 casa {n} vezes, e não 1: {velho.strip()}")
        texto = texto.replace(velho, novo)
    destino.write_text(texto)
    return destino


TABELAS = ("tabela_energia.dat", "tabela_eedf.dat", "tabela_energia_tg.dat", "tabela_energia_tg_conferencia.dat")


def processos_do_bolsigdb(arquivo):
    """{nome normalizado: limiar (eV)} dos processos de excitação, ionização e
    captura do bolsigdb.dat; o nome sem espaços e em maiúsculas, no sentido
    direto (A <-> B vira A->B), como o BOLSIG+ os nomeia."""
    linhas = pathlib.Path(arquivo).read_text().splitlines()
    proc = {}
    for i, l in enumerate(linhas):
        if l.strip() in ("EXCITATION", "IONIZATION", "ATTACHMENT"):
            nome = linhas[i + 1].replace(" ", "").upper().replace("<->", "->")
            try:
                limiar = 0.0 if l.strip() == "ATTACHMENT" else float(linhas[i + 2].split("/")[0].split()[0])
            except (ValueError, IndexError):
                raise ErroCaso(f"{arquivo}: processo {nome} sem a linha do limiar") from None
            proc[nome] = limiar
    return proc


def gera_calor_rapido(pasta):
    """calor_rapido.txt da pasta: os processos do BOLSIG+ cuja energia vira calor
    do gás em picossegundos (bloco calor.processos_rapidos do gas.json, por
    exemplo a excitação rotacional do N2), com o limiar do bolsigdb.dat. Sem a
    lista no gas.json, apaga o arquivo e devolve None."""
    import json
    pasta = pathlib.Path(pasta)
    destino = pasta / "calor_rapido.txt"
    lista = json.loads((pasta / "gas.json").read_text()).get("calor", {}).get("processos_rapidos")
    if not lista:
        destino.unlink(missing_ok=True)
        return None
    proc = processos_do_bolsigdb(pasta / "bolsigdb.dat")
    linhas = []
    for nome in lista:
        chave = nome.replace(" ", "").upper().replace("<->", "->")
        if chave not in proc:
            raise ErroCaso(f"gas.json: processo rápido {chave} não está no bolsigdb.dat")
        linhas.append(f"{chave} {proc[chave]!r}")
    destino.write_text("\n".join(linhas) + "\n")
    return destino


# o bolsig_x86_64.so guarda cada seção de choque em arrays de 1000 posições:
# com tabelas grandes cai com falha de segmentação (N -> N(2D) com 1291 pontos
# no banco de Sun et al. 2022). Medido nas tabelas completas: 1000 e 999 caem
# (999 só acima de ~22 Td), 900, 800 e 700 passam; o limite fica em 900
MAXIMO_PONTOS_BOLSIG = 900


def _tabelas_do_bolsigdb(linhas):
    """(índice da linha do tipo, nome normalizado, início e fim das linhas de
    dados) de cada processo do bolsigdb.dat."""
    tipos = ("ELASTIC", "EFFECTIVE", "MOMENTUM", "EXCITATION", "IONIZATION", "ATTACHMENT", "ROTATION")
    saida, i = [], 0
    while i < len(linhas):
        if linhas[i].strip() in tipos:
            nome = linhas[i + 1].replace(" ", "").upper().replace("<->", "->")
            j = i + 1
            while j < len(linhas) and not linhas[j].startswith("-----"):
                j += 1
            k = j + 1
            while k < len(linhas) and not linhas[k].startswith("-----"):
                k += 1
            saida.append((i, nome, j + 1, k))
            i = k
        i += 1
    return saida


def secoes_grandes(arquivo, maximo=MAXIMO_PONTOS_BOLSIG):
    """{processo: pontos} das seções de choque com mais de maximo pontos."""
    linhas = pathlib.Path(arquivo).read_text().splitlines()
    return {nome: fim - ini for _, nome, ini, fim in _tabelas_do_bolsigdb(linhas) if fim - ini > maximo}


def reduz_secoes(origem, destino, maximo=MAXIMO_PONTOS_BOLSIG):
    """Copia o bolsigdb.dat tirando das seções com mais de maximo pontos os
    pontos que menos mudam a curva, até caber (mantém o primeiro e o último). Devolve
    {processo: {pontos: (antes, depois), erro_max}}, com o erro da
    interpolação linear da reduzida nos pontos retirados, relativo ao máximo
    da seção."""
    import numpy as np
    linhas = pathlib.Path(origem).read_text().splitlines()
    relatorio, novas, pos = {}, [], 0
    for _, nome, ini, fim in _tabelas_do_bolsigdb(linhas):
        dados = linhas[ini:fim]
        if len(dados) <= maximo:
            continue
        pontos = np.array([[float(x) for x in l.split()[:2]] for l in dados])
        # tira, um a um, o ponto interno cuja falta menos muda a curva (desvio
        # vertical para a reta entre os vizinhos), até sobrarem maximo
        manter = np.arange(len(dados))
        while len(manter) > maximo:
            e, s = pontos[manter, 0], pontos[manter, 1]
            w = (e[1:-1] - e[:-2]) / (e[2:] - e[:-2])
            desvio = np.abs(s[1:-1] - ((1.0 - w) * s[:-2] + w * s[2:]))
            manter = np.delete(manter, 1 + int(np.argmin(desvio)))
        e, s = pontos[manter, 0], pontos[manter, 1]
        erro = np.max(np.abs(np.interp(pontos[:, 0], e, s) - pontos[:, 1])) / np.max(np.abs(pontos[:, 1]))
        relatorio[nome] = {"pontos": (len(dados), len(manter)), "erro_max": float(erro)}
        novas += linhas[pos:ini] + [dados[m] for m in manter]
        pos = fim
    novas += linhas[pos:]
    pathlib.Path(destino).write_text("\n".join(novas) + "\n")
    return relatorio


def tira_nao_monotonas(arquivo, coluna_t, coluna_eps):
    """Tira de uma tabela por energia média as linhas cuja energia não passa a
    maior já mantida no mesmo bloco de T_g (coluna_t None: um bloco só). O
    BOLSIG+ refaz a grade de energia entre E/N vizinhos e a energia média pode
    cair um pouco (0,1% perto de 350 Td na amônia); o protótipo inverte a tabela
    pela energia e exige monotonia. Devolve as linhas retiradas."""
    arquivo = pathlib.Path(arquivo)
    saida, retiradas = [], []
    bloco, maior = None, -float("inf")
    for linha in arquivo.read_text().splitlines():
        partes = linha.split()
        if not partes or linha.startswith("#"):
            saida.append(linha)
            continue
        t = partes[coluna_t] if coluna_t is not None else None
        if t != bloco:
            bloco, maior = t, -float("inf")
        eps = float(partes[coluna_eps])
        if eps > maior:
            maior = eps
            saida.append(linha)
        else:
            retiradas.append(linha.strip())
    if retiradas:
        arquivo.write_text("\n".join(saida) + "\n")
    return retiradas


TABELAS_POR_ENERGIA = (("tabela_energia.dat", None, 1), ("tabela_energia_tg.dat", 0, 2),
                       ("tabela_energia_tg_conferencia.dat", 0, 2), ("tabela_calor_tg.dat", 0, 2))


# o BOLSIG+ cai em E/N alto em algumas misturas (25:75 de N2:H2 perto de
# 4300 Td): quando a geração cai acima de E_N_MINIMO_DE_TOPO, o topo da tabela
# vai a FRACAO_DO_TOPO do E/N da queda, até TENTATIVAS_DE_TOPO vezes
E_N_MINIMO_DE_TOPO = 100.0
FRACAO_DO_TOPO = 0.9
TENTATIVAS_DE_TOPO = 4


# as mesmas opções do perfil release do Makefile do repositório de pesquisa
OPCOES_FORTRAN = ["-O2", "-march=native", "-g", "-ffp-contract=off"]


def _roda_gerador(pasta):
    """Compila o gera_tabela_energia.f90 contra o zdplaskin_m.F90 do caso, o
    dvode_f90_m.F90 e o BOLSIG+ (terceiros()) e o roda na pasta; (código, saída)."""
    pasta = pathlib.Path(pasta).resolve()
    t = terceiros()
    with tempfile.TemporaryDirectory() as d:
        return _compila_e_roda(pasta, t, pathlib.Path(d))


def _compila_e_roda(pasta, t, build):
    fc = os.environ.get("FC", "gfortran")
    passos = [[fc, *OPCOES_FORTRAN, "-J", str(build), "-c", str(t["dvode"]), "-o", str(build / "dvode.o")],
              [fc, *OPCOES_FORTRAN, "-J", str(build), "-c", str(pasta / "zdplaskin_m.F90"), "-o", str(build / "zdp.o")],
              [fc, *OPCOES_FORTRAN, "-J", str(build), str(RAIZ / "ferramentas" / "gera_tabela_energia.f90"),
               str(build / "dvode.o"), str(build / "zdp.o"), "-o", str(build / "gera_tabela_energia.x"),
               "-L", str(t["bolsig"].parent), f"-l:{t['bolsig'].name}", f"-Wl,-rpath,{t['bolsig'].parent}"]]
    saida = ""
    for cmd in passos:
        r = subprocess.run(cmd, stdin=subprocess.DEVNULL, capture_output=True, text=True)
        saida += r.stdout + r.stderr
        if r.returncode:
            return r.returncode, saida
    r = subprocess.run([str(build / "gera_tabela_energia.x")], cwd=pasta, stdin=subprocess.DEVNULL,
                       capture_output=True, text=True)
    return r.returncode, saida + r.stdout + r.stderr


def _ponto_da_queda(pasta):
    """(T_g, E/N) em que o gerador estava ao cair, do progresso_tabelas.txt."""
    arq = pasta / "progresso_tabelas.txt"
    if not arq.exists():
        return None
    partes = arq.read_text().split()
    return (float(partes[0]), float(partes[1])) if len(partes) >= 2 else None


def gera_tabelas(pasta, roda=None):
    """Tabelas dos elétrons da pasta pelo BOLSIG+ (minutos). Se o BOLSIG+ cai
    em E/N alto, grava o en_max.txt abaixo do ponto da queda, registra em
    tabelas_topo.txt e gera de novo. roda: o gerador (testes)."""
    pasta = pathlib.Path(pasta).resolve()
    roda = roda or _roda_gerador
    for nome in ("zdplaskin_m.F90", "bolsigdb.dat"):
        if not (pasta / nome).exists():
            raise ErroCaso(f"{pasta}: falta o {nome}")
    grandes = secoes_grandes(pasta / "bolsigdb.dat")
    if grandes:
        raise ErroCaso(f"{pasta / 'bolsigdb.dat'}: seções de choque com mais de {MAXIMO_PONTOS_BOLSIG} pontos, que o "
                       "BOLSIG+ não comporta: " + ", ".join(f"{p} ({n})" for p, n in grandes.items())
                       + f". Use: python3 ferramentas/caso.py reduz {pasta}")
    geradas = TABELAS + ("tabela_calor_tg.dat",)
    esperadas = TABELAS + (("tabela_calor_tg.dat",) if (pasta / "calor_rapido.txt").exists() else ())
    arq_topo = pasta / "en_max.txt"
    topo_original = arq_topo.read_text() if arq_topo.exists() else None
    quedas = []
    for _ in range(TENTATIVAS_DE_TOPO + 1):
        for nome in geradas + ("progresso_tabelas.txt",):     # nem progresso velho de execução interrompida
            (pasta / nome).unlink(missing_ok=True)
        codigo, saida = roda(pasta)
        queda = _ponto_da_queda(pasta)
        (pasta / "progresso_tabelas.txt").unlink(missing_ok=True)
        faltam = [n for n in esperadas if not (pasta / n).exists()]
        if codigo == 0 and not faltam:
            break
        for nome in geradas:                 # tabela parcial não pode passar por pronta
            (pasta / nome).unlink(missing_ok=True)
        # só a queda do BOLSIG+ em E/N alto vira topo novo; qualquer outra falha
        # (compilação, E/N baixo, tentativas esgotadas) devolve o en_max.txt original
        do_bolsig = "BOLSIG" in saida.upper()
        if (queda is None or not do_bolsig or queda[1] <= E_N_MINIMO_DE_TOPO
                or len(quedas) == TENTATIVAS_DE_TOPO):
            if topo_original is None:
                arq_topo.unlink(missing_ok=True)
            else:
                arq_topo.write_text(topo_original)
            raise ErroCaso(f"a geração das tabelas falhou em {pasta}"
                           + (f" (T_g {queda[0]:g} K, E/N {queda[1]:.4g} Td)" if queda else "")
                           + (f"; topos tentados: {', '.join(f'{q[1]:g}' for q in quedas)} Td" if quedas else "")
                           + ":\n" + saida[-3000:])
        topo = float(f"{FRACAO_DO_TOPO * queda[1]:.2g}")
        quedas.append((queda, topo))
        arq_topo.write_text(f"{topo:g}\n")
    if quedas:
        (pasta / "tabelas_topo.txt").write_text(
            "# topo de E/N reduzido porque o BOLSIG+ caiu (ferramentas/caso.py gera_tabelas):\n"
            + (f"# en_max.txt antes: {topo_original.strip()} Td\n" if topo_original else "")
            + "# T_g (K), E/N da queda (Td), novo topo (Td)\n"
            + "".join(f"{q[0]:g} {q[1]:.4g} {topo:g}\n" for q, topo in quedas))
    relatorio = []
    for nome, coluna_t, coluna_eps in TABELAS_POR_ENERGIA:
        if (pasta / nome).exists():
            relatorio += [f"{nome}: {l}" for l in tira_nao_monotonas(pasta / nome, coluna_t, coluna_eps)]
    (pasta / "tabelas_monotonia.txt").unlink(missing_ok=True)
    if relatorio:
        (pasta / "tabelas_monotonia.txt").write_text(
            "# linhas retiradas por energia média não crescente (ruído da grade do BOLSIG+; caso.py tira_nao_monotonas)\n"
            + "\n".join(relatorio) + "\n")
    return [pasta / n for n in TABELAS]


def reduz_bolsigdb(pasta):
    """Na pasta do caso: guarda o bolsigdb.dat como bolsigdb_original.dat (uma
    vez) e grava o reduzido no lugar, com o relatório em reducao_secoes.txt."""
    pasta = pathlib.Path(pasta)
    original, atual = pasta / "bolsigdb_original.dat", pasta / "bolsigdb.dat"
    if original.exists():
        # o bolsigdb.dat tem de ser a redução deste original; senão foi trocado
        # depois, e reduzir o original antigo apagaria o novo
        provisorio = pasta / "bolsigdb_reduzido.tmp"
        try:
            reduz_secoes(original, provisorio)
            igual = provisorio.read_text().rstrip() == atual.read_text().rstrip()
        finally:
            provisorio.unlink(missing_ok=True)
        if not igual:
            raise ErroCaso(f"{atual} não é a redução do {original}: se o bolsigdb.dat é novo, apague o "
                           "bolsigdb_original.dat antigo e rode reduz de novo")
    else:
        atual.rename(original)
    relatorio = reduz_secoes(original, atual)
    texto = ["# seções de choque reduzidas para caber no BOLSIG+ (ferramentas/caso.py reduz):",
             "# processo, pontos antes e depois, erro máximo da interpolação linear relativo ao máximo"]
    texto += [f"{p} {v['pontos'][0]} {v['pontos'][1]} {v['erro_max']:.3e}" for p, v in relatorio.items()]
    (pasta / "reducao_secoes.txt").write_text("\n".join(texto) + "\n")
    return relatorio


def reduz_se_preciso(pasta):
    """Reduz o bolsigdb.dat só se ele tem seções grandes demais para o BOLSIG+;
    devolve o relatório da redução ({} se não precisou)."""
    if not secoes_grandes(pathlib.Path(pasta) / "bolsigdb.dat"):
        return {}
    return reduz_bolsigdb(pasta)


# massas atômicas (u), IUPAC 2021, valores convencionais
MASSAS = {"H": 1.008, "HE": 4.0026, "C": 12.011, "N": 14.007, "O": 15.999, "F": 18.998, "NE": 20.180,
          "SI": 28.085, "S": 32.06, "CL": 35.45, "AR": 39.948, "KR": 83.798, "XE": 131.29}


def massa_u(especie):
    """Massa (u) pela fórmula do nome do ZDPlasKin: sem carga (^+, ^-), sem o
    estado entre parênteses e sem *. Com as letras em maiúsculas, duas letras
    que são elementos separados valem como dois (NO = N + O, CO = C + O); senão,
    como um (AR, HE, CL)."""
    nome = especie.upper().replace("^+", "").replace("^-", "").replace("*", "")
    nome = nome.split("(")[0]
    total, i = 0.0, 0
    while i < len(nome):
        dois, um = nome[i:i + 2], nome[i]
        if len(dois) == 2 and dois.isalpha() and dois in MASSAS and not (um in MASSAS and dois[1] in MASSAS):
            elem, i = dois, i + 2
        elif um in MASSAS:
            elem, i = um, i + 1
        else:
            raise ErroCaso(f"massa de {especie}: elemento desconhecido em {nome[i:]}; escreva massa_u no transporte.json")
        j = i
        while j < len(nome) and nome[j].isdigit():
            j += 1
        total += MASSAS[elem] * (int(nome[i:j]) if j > i else 1)
        i = j
    return total


BIBLIOTECA = RAIZ / "biblioteca" / "transporte.json"

# polarizabilidades (Å3) dos gases de fundo comuns, CRC Handbook of Chemistry
# and Physics, tabela de polarizabilidades atômicas e moleculares; outras vão em
# "polarizabilidades" no gas.json
POLARIZABILIDADES = {"HE": 0.2050, "NE": 0.3956, "AR": 1.6411, "KR": 2.4844, "XE": 4.044, "H2": 0.8045,
                     "N2": 1.7403, "O2": 1.5812, "CO": 1.95, "CO2": 2.911, "CH4": 2.593, "H2O": 1.45}


def mobilidade_estimada(ion, fundo, polarizabilidades=None):
    """Mobilidade reduzida (cm2/(V s)) do íon no gás de fundo {gás: fração}:
    limite de Langevin K0 = 13,853/sqrt(alfa mu) em cada gás (alfa em Å3, mu
    a massa reduzida em u) e lei de Blanc na mistura, 1/K = soma de x_i/K_i.
    É a polarização pura, sem troca de carga: no íon do próprio gás (N2+ em
    N2) a mobilidade medida é menor."""
    alfas = {k.upper(): v for k, v in (polarizabilidades or {}).items()}
    inverso = 0.0
    for gas, fracao in fundo.items():
        alfa = alfas.get(gas.upper(), POLARIZABILIDADES.get(gas.upper()))
        if alfa is None:
            raise ErroCaso(f"polarizabilidade de {gas} desconhecida: ponha-a (Å3) em polarizabilidades no gas.json")
        m1, m2 = massa_u(ion), massa_u(gas)
        inverso += fracao / (13.853 / (alfa * m1 * m2 / (m1 + m2)) ** 0.5)
    return 1.0 / inverso


def difusao_estimada(especie, fundo, _cache={}):
    """Difusão de traço {D0 (m2/s a 1 atm e 300 K), T0, expoente} da espécie no
    fundo {gás: fração} pela teoria cinética (Cantera, GRI-Mech 3.0): 1/D =
    soma x_j/D_kj, expoente ajustado de 250 a 1000 K. Estados excitados e íons
    usam a espécie de base (N2(A3) como N2). None se a espécie ou um gás do
    fundo não está no GRI-Mech."""
    try:
        import cantera as ct
    except ImportError:
        raise ErroCaso("estimar a difusão pede o Cantera (pip install cantera)") from None
    import numpy as np
    if "g" not in _cache:
        _cache["g"] = ct.Solution("gri30.yaml")
        _cache["g"].transport_model = "mixture-averaged"
    g = _cache["g"]
    nomes = {n.upper(): n for n in g.species_names}
    base = especie.upper().replace("^+", "").replace("^-", "").replace("*", "").split("(")[0]
    if base not in nomes or any(f.upper() not in nomes for f in fundo):
        return None
    k = g.species_index(nomes[base])
    x = {nomes[f.upper()]: v for f, v in fundo.items()}
    ts = np.array([250.0, 300.0, 400.0, 600.0, 1000.0])
    d = []
    for temp in ts:
        g.TPX = temp, ct.one_atm, x
        b = g.binary_diff_coeffs
        d.append(1.0 / sum(v / b[k, g.species_index(n)] for n, v in x.items()))
    d = np.array(d)
    expoente = np.polyfit(np.log(ts / 300.0), np.log(d / d[1]), 1)[0]
    return {"D0": float(d[1]), "T0": 300.0, "expoente": round(float(expoente), 3)}


def gera_transporte(pasta, biblioteca=None, quimica=None, estimar=False):
    """transporte.json do caso: preserva o que já existe nele, completa com a
    massa pela fórmula e com a biblioteca (ESPECIE em FUNDO; numa mistura,
    ESPECIE em A+B, na ordem do gas.json, porque o dado num gás só não vale
    para a mistura). Devolve {espécie: [campos que faltam]}: mobilidade dos
    íons, positivos e negativos, difusão e perda na parede dos neutros que não
    são do fundo (atomos_do_fundo só com --gas-solto). quimica: a Quimica já
    lida, no lugar do zdplaskin_m.F90 da pasta. estimar: preenche o que falta
    de mobilidade (Langevin e Blanc) e de difusão (teoria cinética), com a
    fonte marcada como estimativa; a perda na parede não se estima."""
    import json
    sys.path.insert(0, str(RAIZ / "prototipo"))
    from quimica_zdp import Quimica
    pasta = pathlib.Path(pasta)
    q = quimica if quimica is not None else Quimica(pasta / "zdplaskin_m.F90")
    gas = json.loads((pasta / "gas.json").read_text())
    nomes_fundo = [f.upper() for f in gas["fundo"]]
    fundo = "+".join(nomes_fundo)
    if biblioteca is None:
        biblioteca = json.loads(BIBLIOTECA.read_text())
    arq = pasta / "transporte.json"
    atual = json.loads(arq.read_text()) if arq.exists() else {}
    saida = {"_formato": atual.get("_formato", json.loads((RAIZ / "transporte.json").read_text())["_formato"])}
    faltam = {}
    for k, esp in enumerate(q.especies):
        esp = esp.upper()
        if esp == "E":
            continue
        e = dict(atual.get(esp, {}))
        if "massa_u" not in e:
            e["massa_u"] = massa_u(esp)
        for campo, valor in biblioteca.get(f"{esp} em {fundo}", {}).items():
            e.setdefault(campo, valor)
        if estimar:
            _estima(e, esp, q.carga[k], gas, nomes_fundo)
        if q.carga[k] != 0:
            exigidos = ["mobilidade"]
        elif q.carga[k] == 0 and esp not in nomes_fundo:
            exigidos = ["difusao", "perda_na_parede"]
        else:
            exigidos = []
        if [c for c in exigidos if c not in e]:
            faltam[esp] = [c for c in exigidos if c not in e]
        saida[esp] = e
    arq.write_text(json.dumps(saida, ensure_ascii=False, indent=2) + "\n")
    return faltam


def _estima(e, esp, carga, gas, nomes_fundo):
    """Completa a entrada e da espécie com as estimativas que faltam."""
    notas = []
    if carga != 0 and "mobilidade" not in e:
        e["mobilidade"] = {"tipo": "constante",
                           "K0": mobilidade_estimada(esp, gas["fundo"], gas.get("polarizabilidades"))}
        notas.append("mobilidade pelo limite de Langevin com a lei de Blanc (sem troca de carga)")
    # o fundo também, para --gas-solto, onde ele difunde
    if carga == 0 and "difusao" not in e and tem_cantera():
        d = difusao_estimada(esp, gas["fundo"])
        if d is not None:
            e["difusao"] = d
            notas.append("difusão de traço pela teoria cinética (Cantera, GRI-Mech 3.0) da espécie de base")
    if notas:
        nota = "estimativa automática (caso.py estima): " + "; ".join(notas)
        e["fonte"] = f"{e['fonte']}; {nota}" if isinstance(e.get("fonte"), str) else nota


def limiares_do_bolsigdb(arquivo):
    """{(alvo, produto): limiar em eV} dos processos de excitação e ionização
    do bolsigdb.dat, com os nomes normalizados (maiúsculas, sem espaços)."""
    linhas = pathlib.Path(arquivo).read_text().splitlines()
    saida = {}
    for i, nome, _, _ in _tabelas_do_bolsigdb(linhas):
        if linhas[i].strip() not in ("EXCITATION", "IONIZATION") or "->" not in nome:
            continue
        a, b = nome.split("->", 1)
        try:
            saida[(a, b)] = float(linhas[i + 2].split("/")[0].split()[0])
        except (ValueError, IndexError):
            raise ErroCaso(f"{arquivo}: processo {nome} sem a linha do limiar (eV)") from None
    return saida


def propaga_entalpias(q, limiares, conhecidas):
    """Entalpias (eV por partícula) a partir das conhecidas, pelos processos
    do BOLSIG+ do caso com um reagente e um produto pesados (excitação e
    ionização simples): h(produto) = h(reagente) + limiar, com o menor limiar
    quando vários levam ao mesmo produto (o nível mais baixo, como v = 0).
    Devolve (entalpias, origem de cada uma), com os nomes do kinet.inp."""
    import numpy as np
    nome = {e.upper(): e for e in q.especies}
    h = {nome.get(k.upper(), k): float(v) for k, v in conhecidas.items()}
    origem = {k: "dada" for k in h}
    ie = q.indice["E"]
    while True:
        candidatos = {}
        for i, r in enumerate(q.reacoes):
            if q.coluna_bolsig[i] is None:
                continue
            processo = tuple(r.split(":", 1)[1].upper().split("->", 1))
            if processo not in limiares:
                continue
            nu = q.estequiometria[:, i]
            pesados = [s for s in np.nonzero(nu)[0] if s != ie]
            reag = [s for s in pesados if nu[s] < 0]
            prod = [s for s in pesados if nu[s] > 0]
            if len(reag) != 1 or len(prod) != 1 or nu[reag[0]] != -1 or nu[prod[0]] != 1:
                continue
            a, b = q.especies[reag[0]], q.especies[prod[0]]
            if a in h and b not in h:
                valor = h[a] + limiares[processo]
                if b not in candidatos or valor < candidatos[b][0]:
                    candidatos[b] = (valor, f"{a} mais o limiar de {r.split(':', 1)[1]} ({limiares[processo]:g} eV)")
        if not candidatos:
            return h, origem
        for b, (valor, texto) in candidatos.items():
            h[b], origem[b] = valor, texto


EV_POR_J_KMOL = 1.0 / (6.02214076e26 * 1.602176634e-19)


def entalpia_de_formacao(especie):
    """Entalpia de formação a 298,15 K (eV por partícula) da espécie no estado
    fundamental pelo GRI-Mech 3.0 (Cantera); None se não está nele ou se o nome
    é de estado excitado ou íon."""
    if any(c in especie for c in "()*^"):
        return None
    import cantera as ct
    g = ct.Solution("gri30.yaml")
    nomes = {n.upper(): n for n in g.species_names}
    if especie.upper() not in nomes:
        return None
    return g.species(nomes[especie.upper()]).thermo.h(298.15) * EV_POR_J_KMOL


def estima_entalpias(pasta):
    """Completa calor.entalpias no gas.json do caso: as espécies no estado
    fundamental pelo GRI-Mech 3.0 (Cantera) e as demais pelos limiares do
    BOLSIG+ (propaga_entalpias). Preserva o que já existe, registra a origem de
    cada valor novo na fonte do bloco e devolve as espécies que faltam."""
    import json
    sys.path.insert(0, str(RAIZ / "prototipo"))
    from quimica_zdp import Quimica
    pasta = pathlib.Path(pasta)
    q = Quimica(pasta / "zdplaskin_m.F90")
    gas = json.loads((pasta / "gas.json").read_text())
    tinha_calor = "calor" in gas
    calor = gas.setdefault("calor", {})
    ent = calor.setdefault("entalpias", {})
    existentes = {k.upper() for k in ent}
    base = dict(ent)
    novas_origem = {}
    if tem_cantera():
        for esp in q.especies:
            if esp.upper() == "E":
                continue
            h = entalpia_de_formacao(esp)
            if h is None:
                continue
            if esp.upper() not in existentes:
                base[esp] = h
                novas_origem[esp] = "entalpia de formação a 298 K do GRI-Mech 3.0 (Cantera)"
            else:
                dada = next(v for k, v in ent.items() if k.upper() == esp.upper())
                if abs(dada - h) > 0.05:
                    print(f"aviso: {esp} tem {dada:g} eV no gas.json e {h:.3f} eV no GRI-Mech: as entalpias "
                          "novas usam os elementos a 298 K como referência; confira a do gas.json")
    else:
        print("aviso: sem o Cantera (pip install cantera) as espécies no estado fundamental ficam de fora")
    h, origem = propaga_entalpias(q, limiares_do_bolsigdb(pasta / "bolsigdb.dat"), base)
    for esp, valor in h.items():
        if esp.upper() not in existentes:
            ent[esp] = round(valor, 4)
            novas_origem.setdefault(esp, origem[esp])
    com = {k.upper() for k in list(ent) + [e for e in h if e.upper() not in existentes]}
    faltam = [e for e in q.especies if e.upper() != "E" and e.upper() not in com]
    if not tinha_calor and faltam:
        # um bloco calor incompleto faria o modelo recusar o caso: a proposta
        # fica à parte e o gas.json não muda
        proposta = {"entalpias": {e: round(v, 4) for e, v in h.items()},
                    "origem": {e: novas_origem.get(e, origem[e]) for e in h}, "faltam": faltam}
        (pasta / "entalpias_propostas.json").write_text(json.dumps(proposta, ensure_ascii=False, indent=2) + "\n")
        print(f"o gas.json não tem bloco calor e faltam entalpias: a proposta ficou em "
              f"{pasta / 'entalpias_propostas.json'}; complete-a e copie para calor.entalpias")
        return faltam
    if not tinha_calor:
        print("aviso: o bloco calor novo liga o calor das reações no modelo")
    if novas_origem:
        fonte = calor.get("fonte", [])
        fonte = [fonte] if isinstance(fonte, str) else list(fonte)
        fonte.append("estimativa automática (caso.py entalpias), em eV por partícula relativa aos elementos a "
                     "298 K: " + "; ".join(f"{e}: {o}" for e, o in novas_origem.items()))
        calor["fonte"] = fonte
    (pasta / "gas.json").write_text(json.dumps(gas, ensure_ascii=False, indent=2) + "\n")
    return faltam


def tem_cantera():
    try:
        import cantera  # noqa: F401
    except ImportError:
        return False
    return True


def estima_transporte(pasta):
    """transporte.json com as estimativas no que falta (gera_transporte com
    estimar); sem o Cantera, só as mobilidades, e a difusão fica listada."""
    if not tem_cantera():
        print("aviso: sem o Cantera (pip install cantera) a difusão dos neutros não é estimada")
    return gera_transporte(pasta, estimar=True)


def exporta(pasta):
    """quimica.json da pasta a partir do zdplaskin_m.F90 e do bolsigdb.dat: a
    química que o simulador usa sem o código do ZDPlasKin (para distribuir o
    caso, que não pode levar o zdplaskin_m.F90)."""
    pasta = pathlib.Path(pasta)
    sys.path.insert(0, str(RAIZ / "prototipo"))
    from quimica_zdp import Quimica
    for nome in ("zdplaskin_m.F90", "bolsigdb.dat"):
        if not (pasta / nome).exists():
            raise ErroCaso(f"{pasta}: falta o {nome} para exportar a química")
    Quimica(pasta / "zdplaskin_m.F90").salva(pasta / "quimica.json", bolsigdb=pasta / "bolsigdb.dat")
    return pasta / "quimica.json"


ACOES = {"zdplaskin": [gera_zdplaskin], "tabelas": [gera_tabelas], "transporte": [gera_transporte],
         "calor": [gera_calor_rapido], "reduz": [reduz_bolsigdb],
         "estima": [estima_transporte], "entalpias": [estima_entalpias],
         "exporta": [exporta],
         "tudo": [reduz_se_preciso, gera_zdplaskin, gera_calor_rapido, gera_tabelas, gera_transporte, exporta]}


def main():
    if len(sys.argv) != 3 or sys.argv[1] not in ACOES:
        sys.exit(__doc__)
    try:
        for acao in ACOES[sys.argv[1]]:
            r = acao(sys.argv[2])
            if acao in (gera_transporte, estima_transporte):
                if r:
                    sys.exit("transporte.json gravado; faltam dados:\n" + "\n".join(
                        f"  {esp}: {', '.join(c)}" for esp, c in r.items()))
                print("transporte.json completo")
            elif acao is estima_entalpias:
                if r:
                    sys.exit("gas.json gravado; faltam as entalpias de " + ", ".join(r)
                             + " (eV por partícula, no bloco calor)")
                print("entalpias completas")
            else:
                print(r)
    except ErroCaso as e:
        sys.exit(f"erro: {e}")


if __name__ == "__main__":
    main()
