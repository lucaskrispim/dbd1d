# SPDX-License-Identifier: GPL-3.0-or-later
"""Case tool: prepares a gas mixture for the simulator from the files that
ZDPlasKin already uses, without writing code for each gas.

A case folder holds the kinet.inp and bolsigdb.dat of the mixture. The tool
generates zdplaskin_m.F90 with the ZDPlasKin preprocessor (version 2.0a) and
applies the edits listed in EDICOES_ZDPLASKIN: an internal BOLSIG+ table 10 times
denser and two messages suppressed.

The electron tables (tabela_energia.dat, tabela_eedf.dat, tabela_energia_tg.dat
and the check table) come from BOLSIG+ through the generator
ferramentas/gera_tabela_energia.f90, compiled with gfortran (or FC) against the
zdplaskin_m.F90 of the case, with the composition in composicao.txt.

ZDPlasKin and BOLSIG+ are not distributed with the tool (the licence of ZDPlasKin
allows use, not redistribution): ZDPLASKIN_DIR points to the downloaded ZDPlasKin
folder (preprocessor, dvode_f90_m.F90 and bolsig_x86_64.so) and BOLSIG_SO,
optional, to another BOLSIG+ library. Only generating a new case needs them:
caso.py export writes quimica.json (chemistry and thresholds), which lets the
simulator run without zdplaskin_m.F90 and without bolsigdb.dat.

A cross section with more than 900 points crashes BOLSIG+: caso.py reduce keeps
bolsigdb_original.dat, writes the reduced one and records the reduction error in
reducao_secoes.txt; caso.py all reduces automatically when needed. If BOLSIG+
crashes at high E/N, the generation of the tables writes an E/N limit below the
crash point to en_max.txt and retries (tabelas_topo.txt).

caso.py estimate completes transporte.json with estimates marked in the source
field: ion mobility from the Langevin limit with Blanc's law and neutral
diffusion from kinetic theory (Cantera); the wall loss is not estimated.

caso.py enthalpies completes calor.entalpias in gas.json: ground states from
GRI-Mech 3.0 (Cantera), excited states and ions from the BOLSIG+ thresholds.

Usage: python3 ferramentas/caso.py ACTION FOLDER, with ACTION one of
  zdplaskin    generate zdplaskin_m.F90 from kinet.inp
  heat         write calor_rapido.txt (fast heating processes listed in gas.json)
  tables       generate the electron tables with BOLSIG+
  transport    complete transporte.json from the library
  estimate     complete transporte.json with estimates
  enthalpies   complete the enthalpies in gas.json
  reduce       reduce cross sections too large for BOLSIG+
  export       write quimica.json
  all          reduce if needed, zdplaskin, heat, tables, transport, export
(the Portuguese names of earlier versions, calor, tabelas, transporte, estima,
entalpias, reduz, exporta and tudo, are still accepted)
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
    """Incomplete or invalid case folder."""


def terceiros():
    """Paths of the preprocessor and of dvode_f90_m.F90 of ZDPlasKin and of the
    BOLSIG+ library, which are not in the distribution (the licence of ZDPlasKin
    allows use, not redistribution): ZDPLASKIN_DIR, the downloaded ZDPlasKin
    folder, and BOLSIG_SO, the library (default: bolsig_x86_64.so inside
    ZDPLASKIN_DIR). Without the variables, local copies, if they exist."""
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
        raise ErroCaso("third-party programs are missing: " + ", ".join(faltam) + ". Download ZDPlasKin "
                       "(www.zdplaskin.laplace.univ-tlse.fr), which brings the preprocessor, dvode_f90_m.F90 and "
                       "BOLSIG+, and set its folder in ZDPLASKIN_DIR (and the library in BOLSIG_SO, if another)")
    return caminhos


def gera_zdplaskin(pasta):
    """zdplaskin_m.F90 of the folder from kinet.inp, with the listed edits."""
    pasta = pathlib.Path(pasta)
    destino = pasta / "zdplaskin_m.F90"
    if not (pasta / "kinet.inp").exists():
        raise ErroCaso(f"{pasta}: kinet.inp is missing")
    destino.unlink(missing_ok=True)
    r = subprocess.run([str(terceiros()["preprocessor"]), "kinet.inp", "zdplaskin_m.F90"], cwd=pasta,
                       stdin=subprocess.DEVNULL, capture_output=True, text=True)
    if r.returncode != 0 or not destino.exists() or "ERROR" in r.stdout + r.stderr:
        raise ErroCaso(f"the ZDPlasKin preprocessor refused {pasta / 'kinet.inp'}:\n"
                       + (r.stdout + r.stderr)[-2000:])
    texto = destino.read_text()
    for velho, novo in EDICOES_ZDPLASKIN:
        n = texto.count(velho)
        if n != 1:
            destino.unlink()
            raise ErroCaso(f"edit of zdplaskin_m.F90 matches {n} times, not 1: {velho.strip()}")
        texto = texto.replace(velho, novo)
    destino.write_text(texto)
    return destino


TABELAS = ("tabela_energia.dat", "tabela_eedf.dat", "tabela_energia_tg.dat", "tabela_energia_tg_conferencia.dat")


def processos_do_bolsigdb(arquivo):
    """{normalized name: threshold (eV)} of the excitation, ionization and
    attachment processes of bolsigdb.dat; the name without spaces and in upper
    case, in the forward direction (A <-> B becomes A->B), as BOLSIG+ names them."""
    linhas = pathlib.Path(arquivo).read_text().splitlines()
    proc = {}
    for i, l in enumerate(linhas):
        if l.strip() in ("EXCITATION", "IONIZATION", "ATTACHMENT"):
            nome = linhas[i + 1].replace(" ", "").upper().replace("<->", "->")
            try:
                limiar = 0.0 if l.strip() == "ATTACHMENT" else float(linhas[i + 2].split("/")[0].split()[0])
            except (ValueError, IndexError):
                raise ErroCaso(f"{arquivo}: process {nome} without the threshold line") from None
            proc[nome] = limiar
    return proc


def gera_calor_rapido(pasta):
    """calor_rapido.txt of the folder: the BOLSIG+ processes whose energy becomes
    gas heat within picoseconds (block calor.processos_rapidos of gas.json, for
    example the rotational excitation of N2), with the threshold of bolsigdb.dat.
    Without the list in gas.json, deletes the file and returns None."""
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
            raise ErroCaso(f"gas.json: fast process {chave} is not in bolsigdb.dat")
        linhas.append(f"{chave} {proc[chave]!r}")
    destino.write_text("\n".join(linhas) + "\n")
    return destino


# bolsig_x86_64.so stores each cross section in arrays of 1000 elements: with
# large tables it crashes with a segmentation fault (N -> N(2D) with 1291 points
# in the data of Sun et al. 2022). Measured with the full tables: 1000 and 999
# crash (999 only above ~22 Td), 900, 800 and 700 pass; the limit is 900
MAXIMO_PONTOS_BOLSIG = 900


def _tabelas_do_bolsigdb(linhas):
    """(index of the line with the type, normalized name, start and end of the
    data lines) of each process of bolsigdb.dat."""
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
    """{process: points} of the cross sections with more than maximo points."""
    linhas = pathlib.Path(arquivo).read_text().splitlines()
    return {nome: fim - ini for _, nome, ini, fim in _tabelas_do_bolsigdb(linhas) if fim - ini > maximo}


def reduz_secoes(origem, destino, maximo=MAXIMO_PONTOS_BOLSIG):
    """Copies bolsigdb.dat removing from the cross sections with more than maximo
    points the points that change the curve least, until they fit (keeps the
    first and the last). Returns {process: {pontos: (before, after), erro_max}},
    with the error of the linear interpolation of the reduced section at the
    removed points, relative to the maximum of the section."""
    import numpy as np
    linhas = pathlib.Path(origem).read_text().splitlines()
    relatorio, novas, pos = {}, [], 0
    for _, nome, ini, fim in _tabelas_do_bolsigdb(linhas):
        dados = linhas[ini:fim]
        if len(dados) <= maximo:
            continue
        pontos = np.array([[float(x) for x in l.split()[:2]] for l in dados])
        # removes, one by one, the interior point whose absence changes the curve
        # least (vertical deviation from the line between its neighbours), until
        # maximo remain
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
    """Removes from a table in mean energy the lines whose energy does not exceed
    the largest kept so far in the same T_g block (coluna_t None: a single
    block). BOLSIG+ remakes its energy grid between neighbouring E/N and the mean
    energy can drop slightly (0.1% near 350 Td in the ammonia case); the
    simulator inverts the table by the energy and requires monotonicity. Returns
    the removed lines."""
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


# BOLSIG+ crashes at high E/N in some mixtures (N2:H2 25:75 near 4300 Td): when
# the generation crashes above E_N_MINIMO_DE_TOPO, the top of the table goes to
# FRACAO_DO_TOPO of the E/N of the crash, up to TENTATIVAS_DE_TOPO times
E_N_MINIMO_DE_TOPO = 100.0
FRACAO_DO_TOPO = 0.9
TENTATIVAS_DE_TOPO = 4


# the release profile of the original Makefile without -march=native, so that the tables do not depend
# on the machine (with -ffp-contract=off they come out identical, checked in test_tabelas_do_argonio)
OPCOES_FORTRAN = ["-O2", "-g", "-ffp-contract=off"]


def _roda_gerador(pasta):
    """Compiles gera_tabela_energia.f90 against the zdplaskin_m.F90 of the case,
    dvode_f90_m.F90 and BOLSIG+ (terceiros()) and runs it in the folder; (code, output)."""
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
    """(T_g, E/N) at which the generator was when it crashed, from progresso_tabelas.txt."""
    arq = pasta / "progresso_tabelas.txt"
    if not arq.exists():
        return None
    partes = arq.read_text().split()
    return (float(partes[0]), float(partes[1])) if len(partes) >= 2 else None


def gera_tabelas(pasta, roda=None):
    """Electron tables of the folder from BOLSIG+ (minutes). If BOLSIG+ crashes at
    high E/N, writes en_max.txt below the crash point, records it in
    tabelas_topo.txt and generates again. roda: the generator (tests)."""
    pasta = pathlib.Path(pasta).resolve()
    roda = roda or _roda_gerador
    for nome in ("zdplaskin_m.F90", "bolsigdb.dat"):
        if not (pasta / nome).exists():
            raise ErroCaso(f"{pasta}: {nome} is missing")
    grandes = secoes_grandes(pasta / "bolsigdb.dat")
    if grandes:
        raise ErroCaso(f"{pasta / 'bolsigdb.dat'}: cross sections with more than {MAXIMO_PONTOS_BOLSIG} points, which "
                       "BOLSIG+ cannot hold: " + ", ".join(f"{p} ({n})" for p, n in grandes.items())
                       + f". Use: python3 ferramentas/caso.py reduce {pasta}")
    geradas = TABELAS + ("tabela_calor_tg.dat",)
    esperadas = TABELAS + (("tabela_calor_tg.dat",) if (pasta / "calor_rapido.txt").exists() else ())
    arq_topo = pasta / "en_max.txt"
    topo_original = arq_topo.read_text() if arq_topo.exists() else None
    quedas = []
    for _ in range(TENTATIVAS_DE_TOPO + 1):
        for nome in geradas + ("progresso_tabelas.txt",):     # nor old progress of an interrupted run
            (pasta / nome).unlink(missing_ok=True)
        codigo, saida = roda(pasta)
        queda = _ponto_da_queda(pasta)
        (pasta / "progresso_tabelas.txt").unlink(missing_ok=True)
        faltam = [n for n in esperadas if not (pasta / n).exists()]
        if codigo == 0 and not faltam:
            break
        for nome in geradas:                 # a partial table must not pass as finished
            (pasta / nome).unlink(missing_ok=True)
        # only a crash of BOLSIG+ at high E/N gives a new top; any other failure
        # (compilation, low E/N, attempts exhausted) restores the original en_max.txt
        do_bolsig = "BOLSIG" in saida.upper()
        if (queda is None or not do_bolsig or queda[1] <= E_N_MINIMO_DE_TOPO
                or len(quedas) == TENTATIVAS_DE_TOPO):
            if topo_original is None:
                arq_topo.unlink(missing_ok=True)
            else:
                arq_topo.write_text(topo_original)
            raise ErroCaso(f"the generation of the tables failed in {pasta}"
                           + (f" (T_g {queda[0]:g} K, E/N {queda[1]:.4g} Td)" if queda else "")
                           + (f"; tops tried: {', '.join(f'{q[1]:g}' for q in quedas)} Td" if quedas else "")
                           + ":\n" + saida[-3000:])
        topo = float(f"{FRACAO_DO_TOPO * queda[1]:.2g}")
        quedas.append((queda, topo))
        arq_topo.write_text(f"{topo:g}\n")
    if quedas:
        (pasta / "tabelas_topo.txt").write_text(
            "# top of E/N lowered because BOLSIG+ crashed (ferramentas/caso.py gera_tabelas):\n"
            + (f"# en_max.txt before: {topo_original.strip()} Td\n" if topo_original else "")
            + "# T_g (K), E/N of the crash (Td), new top (Td)\n"
            + "".join(f"{q[0]:g} {q[1]:.4g} {topo:g}\n" for q, topo in quedas))
    relatorio = []
    for nome, coluna_t, coluna_eps in TABELAS_POR_ENERGIA:
        if (pasta / nome).exists():
            relatorio += [f"{nome}: {l}" for l in tira_nao_monotonas(pasta / nome, coluna_t, coluna_eps)]
    (pasta / "tabelas_monotonia.txt").unlink(missing_ok=True)
    if relatorio:
        (pasta / "tabelas_monotonia.txt").write_text(
            "# lines removed for a non-increasing mean energy (noise of the BOLSIG+ grid; caso.py tira_nao_monotonas)\n"
            + "\n".join(relatorio) + "\n")
    return [pasta / n for n in TABELAS]


def reduz_bolsigdb(pasta):
    """In the case folder: keeps bolsigdb.dat as bolsigdb_original.dat (once) and
    writes the reduced one in its place, with the report in reducao_secoes.txt."""
    pasta = pathlib.Path(pasta)
    original, atual = pasta / "bolsigdb_original.dat", pasta / "bolsigdb.dat"
    if original.exists():
        # bolsigdb.dat must be the reduction of this original; otherwise it was
        # replaced afterwards, and reducing the old original would erase the new one
        provisorio = pasta / "bolsigdb_reduzido.tmp"
        try:
            reduz_secoes(original, provisorio)
            igual = provisorio.read_text().rstrip() == atual.read_text().rstrip()
        finally:
            provisorio.unlink(missing_ok=True)
        if not igual:
            raise ErroCaso(f"{atual} is not the reduction of {original}: if bolsigdb.dat is new, delete the old "
                           "bolsigdb_original.dat and run reduce again")
    else:
        atual.rename(original)
    relatorio = reduz_secoes(original, atual)
    texto = ["# cross sections reduced to fit BOLSIG+ (ferramentas/caso.py reduce):",
             "# process, points before and after, maximum error of the linear interpolation relative to the maximum"]
    texto += [f"{p} {v['pontos'][0]} {v['pontos'][1]} {v['erro_max']:.3e}" for p, v in relatorio.items()]
    (pasta / "reducao_secoes.txt").write_text("\n".join(texto) + "\n")
    return relatorio


def reduz_se_preciso(pasta):
    """Reduces bolsigdb.dat only if it has sections too large for BOLSIG+;
    returns the report of the reduction ({} if not needed)."""
    if not secoes_grandes(pathlib.Path(pasta) / "bolsigdb.dat"):
        return {}
    return reduz_bolsigdb(pasta)


# atomic masses (u), IUPAC 2021, conventional values
MASSAS = {"H": 1.008, "HE": 4.0026, "C": 12.011, "N": 14.007, "O": 15.999, "F": 18.998, "NE": 20.180,
          "SI": 28.085, "S": 32.06, "CL": 35.45, "AR": 39.948, "KR": 83.798, "XE": 131.29}


def massa_u(especie):
    """Mass (u) from the formula in the ZDPlasKin name: without charge (^+, ^-),
    without the state in parentheses and without *. With the letters in upper
    case, two letters that are separate elements count as two (NO = N + O, CO =
    C + O); otherwise as one (AR, HE, CL)."""
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
            raise ErroCaso(f"mass of {especie}: unknown element in {nome[i:]}; write massa_u in transporte.json")
        j = i
        while j < len(nome) and nome[j].isdigit():
            j += 1
        total += MASSAS[elem] * (int(nome[i:j]) if j > i else 1)
        i = j
    return total


BIBLIOTECA = RAIZ / "biblioteca" / "transporte.json"

# polarizabilities (A3) of common background gases, CRC Handbook of Chemistry
# and Physics, table of atomic and molecular polarizabilities; others go in
# "polarizabilidades" in gas.json
POLARIZABILIDADES = {"HE": 0.2050, "NE": 0.3956, "AR": 1.6411, "KR": 2.4844, "XE": 4.044, "H2": 0.8045,
                     "N2": 1.7403, "O2": 1.5812, "CO": 1.95, "CO2": 2.911, "CH4": 2.593, "H2O": 1.45}


def mobilidade_estimada(ion, fundo, polarizabilidades=None):
    """Reduced mobility (cm2/(V s)) of the ion in the background gas {gas:
    fraction}: Langevin limit K0 = 13.853/sqrt(alpha mu) in each gas (alpha in
    A3, mu the reduced mass in u) and Blanc's law in the mixture, 1/K = sum of
    x_i/K_i. Pure polarization, without charge exchange: for the ion of the gas
    itself (N2+ in N2) the measured mobility is lower."""
    alfas = {k.upper(): v for k, v in (polarizabilidades or {}).items()}
    inverso = 0.0
    for gas, fracao in fundo.items():
        alfa = alfas.get(gas.upper(), POLARIZABILIDADES.get(gas.upper()))
        if alfa is None:
            raise ErroCaso(f"polarizability of {gas} unknown: put it (A3) in polarizabilidades in gas.json")
        m1, m2 = massa_u(ion), massa_u(gas)
        inverso += fracao / (13.853 / (alfa * m1 * m2 / (m1 + m2)) ** 0.5)
    return 1.0 / inverso


def difusao_estimada(especie, fundo, _cache={}):
    """Trace diffusion {D0 (m2/s at 1 atm and 300 K), T0, expoente} of the species
    in the background {gas: fraction} from kinetic theory (Cantera, GRI-Mech
    3.0): 1/D = sum x_j/D_kj, exponent fitted from 250 to 1000 K. Excited states
    and ions use the base species (N2(A3) as N2). None if the species or a
    background gas is not in GRI-Mech."""
    try:
        import cantera as ct
    except ImportError:
        raise ErroCaso("estimating the diffusion needs Cantera (pip install cantera)") from None
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
    """transporte.json of the case: keeps what it already has, completes it with
    the mass from the formula and with the library (keys "SPECIES em
    BACKGROUND"; in a mixture, "SPECIES em A+B", in the order of gas.json, since
    data for a single gas do not hold for the mixture). Returns {species:
    [missing fields]}: mobility of the ions, positive and negative, diffusion and
    wall loss of the neutrals that are not background (atomos_do_fundo only with
    --free-gas). quimica: an already read Quimica, instead of the
    zdplaskin_m.F90 of the folder. estimar: fills in missing mobilities
    (Langevin and Blanc) and diffusion (kinetic theory), with the source marked
    as an estimate; the wall loss is not estimated."""
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
    """Completes the entry e of the species with the missing estimates."""
    notas = []
    if carga != 0 and "mobilidade" not in e:
        e["mobilidade"] = {"tipo": "constante",
                           "K0": mobilidade_estimada(esp, gas["fundo"], gas.get("polarizabilidades"))}
        notas.append("mobility from the Langevin limit with Blanc's law (no charge exchange)")
    # the background too, for --free-gas, where it diffuses
    if carga == 0 and "difusao" not in e and tem_cantera():
        d = difusao_estimada(esp, gas["fundo"])
        if d is not None:
            e["difusao"] = d
            notas.append("trace diffusion from kinetic theory (Cantera, GRI-Mech 3.0) of the base species")
    if notas:
        nota = "automatic estimate (caso.py estimate): " + "; ".join(notas)
        e["fonte"] = f"{e['fonte']}; {nota}" if isinstance(e.get("fonte"), str) else nota


def limiares_do_bolsigdb(arquivo):
    """{(target, product): threshold in eV} of the excitation and ionization
    processes of bolsigdb.dat, with normalized names (upper case, no spaces)."""
    linhas = pathlib.Path(arquivo).read_text().splitlines()
    saida = {}
    for i, nome, _, _ in _tabelas_do_bolsigdb(linhas):
        if linhas[i].strip() not in ("EXCITATION", "IONIZATION") or "->" not in nome:
            continue
        a, b = nome.split("->", 1)
        try:
            saida[(a, b)] = float(linhas[i + 2].split("/")[0].split()[0])
        except (ValueError, IndexError):
            raise ErroCaso(f"{arquivo}: process {nome} without the threshold line (eV)") from None
    return saida


def propaga_entalpias(q, limiares, conhecidas):
    """Enthalpies (eV per particle) from the known ones, through the BOLSIG+
    processes of the case with one heavy reactant and one heavy product (simple
    excitation and ionization): h(product) = h(reactant) + threshold, with the
    lowest threshold when several lead to the same product (the lowest level,
    such as v = 0). Returns (enthalpies, origin of each), with the names of kinet.inp."""
    import numpy as np
    nome = {e.upper(): e for e in q.especies}
    h = {nome.get(k.upper(), k): float(v) for k, v in conhecidas.items()}
    origem = {k: "given" for k in h}
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
                    candidatos[b] = (valor, f"{a} plus the threshold of {r.split(':', 1)[1]} ({limiares[processo]:g} eV)")
        if not candidatos:
            return h, origem
        for b, (valor, texto) in candidatos.items():
            h[b], origem[b] = valor, texto


EV_POR_J_KMOL = 1.0 / (6.02214076e26 * 1.602176634e-19)


def entalpia_de_formacao(especie):
    """Enthalpy of formation at 298.15 K (eV per particle) of the species in the
    ground state from GRI-Mech 3.0 (Cantera); None if it is not there or if the
    name is that of an excited state or an ion."""
    if any(c in especie for c in "()*^"):
        return None
    import cantera as ct
    g = ct.Solution("gri30.yaml")
    nomes = {n.upper(): n for n in g.species_names}
    if especie.upper() not in nomes:
        return None
    return g.species(nomes[especie.upper()]).thermo.h(298.15) * EV_POR_J_KMOL


def estima_entalpias(pasta):
    """Completes calor.entalpias in the gas.json of the case: ground state species
    from GRI-Mech 3.0 (Cantera) and the others from the BOLSIG+ thresholds
    (propaga_entalpias). Keeps what exists, records the origin of each new value
    in the source of the block and returns the missing species."""
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
                novas_origem[esp] = "enthalpy of formation at 298 K from GRI-Mech 3.0 (Cantera)"
            else:
                dada = next(v for k, v in ent.items() if k.upper() == esp.upper())
                if abs(dada - h) > 0.05:
                    print(f"warning: {esp} has {dada:g} eV in gas.json and {h:.3f} eV in GRI-Mech: the new "
                          "enthalpies use the elements at 298 K as reference; check that of gas.json")
    else:
        print("warning: without Cantera (pip install cantera) the ground state species are left out")
    h, origem = propaga_entalpias(q, limiares_do_bolsigdb(pasta / "bolsigdb.dat"), base)
    for esp, valor in h.items():
        if esp.upper() not in existentes:
            ent[esp] = round(valor, 4)
            novas_origem.setdefault(esp, origem[esp])
    com = {k.upper() for k in list(ent) + [e for e in h if e.upper() not in existentes]}
    faltam = [e for e in q.especies if e.upper() != "E" and e.upper() not in com]
    if not tinha_calor and faltam:
        # an incomplete heat block would make the model refuse the case: the
        # proposal is kept apart and gas.json does not change
        proposta = {"entalpias": {e: round(v, 4) for e, v in h.items()},
                    "origem": {e: novas_origem.get(e, origem[e]) for e in h}, "faltam": faltam}
        (pasta / "entalpias_propostas.json").write_text(json.dumps(proposta, ensure_ascii=False, indent=2) + "\n")
        print(f"gas.json has no heat block and enthalpies are missing: the proposal is in "
              f"{pasta / 'entalpias_propostas.json'}; complete it and copy it to calor.entalpias")
        return faltam
    if not tinha_calor:
        print("warning: the new heat block switches on the reaction heat in the model")
    if novas_origem:
        fonte = calor.get("fonte", [])
        fonte = [fonte] if isinstance(fonte, str) else list(fonte)
        fonte.append("automatic estimate (caso.py enthalpies), in eV per particle relative to the elements at "
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
    """transporte.json with estimates for what is missing (gera_transporte with
    estimar); without Cantera, only the mobilities, and the diffusion is listed."""
    if not tem_cantera():
        print("warning: without Cantera (pip install cantera) the diffusion of the neutrals is not estimated")
    return gera_transporte(pasta, estimar=True)


def exporta(pasta):
    """quimica.json of the folder from zdplaskin_m.F90 and bolsigdb.dat: the
    chemistry the simulator uses without the code of ZDPlasKin (to distribute
    the case, which cannot carry zdplaskin_m.F90)."""
    pasta = pathlib.Path(pasta)
    sys.path.insert(0, str(RAIZ / "prototipo"))
    from quimica_zdp import Quimica
    for nome in ("zdplaskin_m.F90", "bolsigdb.dat"):
        if not (pasta / nome).exists():
            raise ErroCaso(f"{pasta}: {nome} is missing to export the chemistry")
    Quimica(pasta / "zdplaskin_m.F90").salva(pasta / "quimica.json", bolsigdb=pasta / "bolsigdb.dat")
    return pasta / "quimica.json"


ACOES = {"zdplaskin": [gera_zdplaskin], "tabelas": [gera_tabelas], "transporte": [gera_transporte],
         "calor": [gera_calor_rapido], "reduz": [reduz_bolsigdb],
         "estima": [estima_transporte], "entalpias": [estima_entalpias],
         "exporta": [exporta],
         "tudo": [reduz_se_preciso, gera_zdplaskin, gera_calor_rapido, gera_tabelas, gera_transporte, exporta]}
# English names of the actions (the Portuguese ones above are still accepted)
ACOES.update({"heat": ACOES["calor"], "tables": ACOES["tabelas"], "transport": ACOES["transporte"],
              "estimate": ACOES["estima"], "enthalpies": ACOES["entalpias"], "reduce": ACOES["reduz"],
              "export": ACOES["exporta"], "all": ACOES["tudo"]})


def main():
    if len(sys.argv) == 2 and sys.argv[1] in ("-h", "--help"):
        print(__doc__)
        return
    if len(sys.argv) != 3 or sys.argv[1] not in ACOES:
        sys.exit(__doc__)
    sys.path.insert(0, str(RAIZ / "prototipo"))
    from quimica_zdp import ErroQuimica
    try:
        for acao in ACOES[sys.argv[1]]:
            r = acao(sys.argv[2])
            if acao in (gera_transporte, estima_transporte):
                if r:
                    sys.exit("transporte.json written; data are missing:\n" + "\n".join(
                        f"  {esp}: {', '.join(c)}" for esp, c in r.items()))
                print("transporte.json complete")
            elif acao is estima_entalpias:
                if r:
                    sys.exit("gas.json written; the enthalpies of " + ", ".join(r)
                             + " are missing (eV per particle, in the calor block)")
                print("enthalpies complete")
            else:
                print(r)
    except (ErroCaso, ErroQuimica) as e:
        sys.exit(f"error: {e}")


if __name__ == "__main__":
    main()
