# SPDX-License-Identifier: GPL-3.0-or-later
"""What each case really loads into BOLSIG+: the blocks of its bolsigdb.dat whose
names appear in its bolsigdb.log (the list of processes BOLSIG+ loaded)."""
import ast
import hashlib
import os
import pathlib
import re
import subprocess
import sys
from dataclasses import dataclass

import numpy as np

RAIZ = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from acoplamento_vibracional import blocos as _separa_blocos  # noqa: E402

MOMENTO = ("ELASTIC", "EFFECTIVE")


@dataclass
class Bloco:
    nome: str            # target line, e.g. "N2 -> N2(SUM)"
    alvo: str            # first species of the target line, e.g. "N2"
    tipo: str            # EXCITATION, IONIZATION, ELASTIC, EFFECTIVE, ROTATION, ATTACHMENT
    limiar: object       # threshold (eV) or None
    razao_pesos: object  # statistical weight ratio (only with <->) or None
    processo: str        # text of the PROCESS: line, or ""
    secao: np.ndarray    # N x 2 array, energy (eV) and cross section (m2)


def _numero(texto):
    return float(texto.replace("d", "e").replace("D", "e"))


def _bloco(linhas):
    tipo = linhas[0].strip()
    nome = linhas[1].strip()
    alvo = re.split(r"<->|->", nome)[0].strip()
    limiar = razao = None
    campos = linhas[2].split("/")[0].split() if len(linhas) > 2 else []
    # for ELASTIC and EFFECTIVE that line is the mass ratio, not a threshold
    if campos and tipo not in MOMENTO and re.match(r"\s*[-+]?[\d.]", linhas[2]):
        try:
            limiar = _numero(campos[0])
            if "<->" in nome and len(campos) > 1:
                razao = _numero(campos[1])
        except ValueError:
            limiar = None
    processo = ""
    for l in linhas[2:]:
        if l.startswith("PROCESS:"):
            processo = l[len("PROCESS:"):].strip()
    tracos = [i for i, l in enumerate(linhas) if l.startswith("-----")]
    dados = [[_numero(x) for x in l.split()[:2]] for l in linhas[tracos[0] + 1:tracos[1]] if l.strip()]
    return Bloco(nome, alvo, tipo, limiar, razao, processo, np.array(dados, dtype=float).reshape(-1, 2))


def blocos_de_texto(texto):
    """Blocos of a bolsigdb.dat given as text (e.g. a coupled bank not yet written)."""
    return [_bloco(l) for k, l in _separa_blocos(texto) if k == "bloco"]


def _normaliza(nome):
    return nome.replace(" ", "").upper().replace("<->", "->")


def todos_os_blocos(pasta):
    """All the blocks of bolsigdb.dat of the folder."""
    return blocos_de_texto((pathlib.Path(pasta) / "bolsigdb.dat").read_text())


def _log(pasta):
    return [l.strip() for l in (pathlib.Path(pasta) / "bolsigdb.log").read_text().splitlines()
            if l.strip() and not l.startswith("#")]


def _especie_do_momento(linha):
    """Species of a momentum transfer log line (C1_N2_Elastic), else None. BOLSIG+ prints
    C*** in place of the counter once it passes 999."""
    m = re.match(r"C(?:\d+|\*+)_(.+?)_(Elastic|Effective)", linha)
    return m.group(1).upper() if m else None


def _casa(pasta):
    """(matched blocks in log order, log lines without a block, superelastic
    lines). A superelastic line (A*->A) is the inverse BOLSIG+ creates for a
    block written with <->, so it has no block of its own."""
    blocos = todos_os_blocos(pasta)
    por_nome, momento = {}, {}
    for b in blocos:
        if b.tipo not in MOMENTO:
            por_nome.setdefault(_normaliza(b.nome), []).append(b)
    for b in blocos:
        if b.tipo in MOMENTO:
            momento.setdefault(b.alvo.upper(), []).append(b)
    inversos = {}
    for b in blocos:
        if "<->" in b.nome:
            a, c = _normaliza(b.nome).split("->", 1)
            inversos[f"{c}->{a}"] = b
    achados, sem, super_ = [], [], []
    for linha in _log(pasta):
        esp = _especie_do_momento(linha)
        fila = momento.get(esp) if esp else por_nome.get(linha.replace(" ", "").upper())
        b = fila.pop(0) if fila else None  # the Nth log line takes the Nth block of that name
        if b is not None:
            achados.append(b)
        elif linha.replace(" ", "").upper() in inversos:
            super_.append(linha)
        else:
            sem.append(linha)
    return achados, sem, super_


def blocos_carregados(pasta):
    """Blocks of bolsigdb.dat whose names appear in bolsigdb.log, in log order."""
    return _casa(pasta)[0]


def nao_casados(pasta):
    """Log lines with no block in bolsigdb.dat and no <-> block to explain them."""
    return _casa(pasta)[1]


def superelasticos(pasta):
    """Log lines that are the superelastic inverse of a <-> block."""
    return _casa(pasta)[2]


def pasta_do_caso(caso):
    return RAIZ if caso == "." else RAIZ / "casos" / caso


# Case names are relative to casos/ and may be nested (n2_amonia_superelastica/tv2000).
CASOS = sorted(p.parent.relative_to(RAIZ / "casos").as_posix() for p in (RAIZ / "casos").glob("**/bolsigdb.log"))
if (RAIZ / "bolsigdb.log").exists():
    CASOS.append(".")


def referencia_transporte(especie, fundo):
    """Key of biblioteca/transporte.json that holds the transport data of the
    species in the given background, or None when the library has no entry that
    applies. Only a pure N2 background is covered: the background N2 itself and the N2
    vibrational states take "N2 em N2", N, N(2D) and N(2P) take their own entry, N2(a'1), N2(B)
    and N2(C) take their own entry, the other N2 electronic states take "N2(A) em N2", and the ions N+,
    N2+, N3+, N4+ take their own "X em N2" entry."""
    if [k.upper() for k in fundo] != ["N2"]:
        return None
    s = especie.strip().upper()
    if s == "N2" or re.fullmatch(r"N2\(V\d+\)", s):
        return "N2 em N2"
    if s == "N":
        return "N em N2"
    if re.fullmatch(r"N\((2D|2P)\)", s):
        return f"{s} em N2"
    if re.fullmatch(r"N2\((A'1?|A`1)\)", s):
        return "N2(A') em N2"
    if re.fullmatch(r"N2\((B|C)3?\)", s):
        return f"N2({s[3]}) em N2"
    if re.fullmatch(r"N2\((A3?|A``1|A''1)\)", s):
        return "N2(A) em N2"
    if s in ("N^+", "N2^+", "N3^+", "N4^+"):
        return f"{s} em N2"
    return None


def _mesma_secao(a, b, tolerancia=0.01):
    """True when two cross sections agree within the relative tolerance at every
    energy point they have in common. A point of one grid counts as common when
    the other grid has a point within 1e-6 relative of it; the other section is
    read there by linear interpolation. Without at least two common points the
    sections are compared at every point of both grids inside the common range."""
    lo = max(a[0, 0], b[0, 0])
    hi = min(a[-1, 0], b[-1, 0])
    if hi < lo:
        return False
    ea = a[:, 0][(a[:, 0] >= lo) & (a[:, 0] <= hi)]
    eb = b[:, 0][(b[:, 0] >= lo) & (b[:, 0] <= hi)]
    e = np.array([x for x in ea if np.any(np.abs(eb - x) <= 1e-6 * max(abs(x), 1e-12))])
    if len(e) < 2:
        e = np.unique(np.concatenate([ea, eb]))
    sa = np.interp(e, a[:, 0], a[:, 1])
    sb = np.interp(e, b[:, 0], b[:, 1])
    escala = np.maximum(np.abs(sa), np.abs(sb))
    return bool(np.all(np.abs(sa - sb) <= tolerancia * escala))


def _limiares_proximos(a, b, tolerancia=0.05):
    if a is None or b is None:
        return a is None and b is None
    return abs(a - b) < tolerancia


def duplicatas(blocos):
    """Pairs (name, name) of blocks with the same target and type, thresholds
    less than 0.05 eV apart and cross sections equal within 1%: the same
    process loaded twice, which BOLSIG+ counts twice in the energy balance."""
    pares = []
    for i, a in enumerate(blocos):
        for b in blocos[i + 1:]:
            if (a.alvo == b.alvo and a.tipo == b.tipo and _limiares_proximos(a.limiar, b.limiar)
                    and len(a.secao) and len(b.secao) and _mesma_secao(a.secao, b.secao)):
                pares.append((a.nome, b.nome))
    return pares


_ATOMO = re.compile(r"[A-Z][a-z]?(\(.*\))?$")   # element symbol, optional state, no digit outside it


def _produtos(lado):
    return [p.strip() for p in re.split(r"\s\+\s|(?<=[\w)])\+(?=[A-Z])", lado) if p.strip()]


def _atomos(produtos):
    return [p for p in produtos if p.upper() != "E" and _ATOMO.match(p.replace(" ", ""))]


def rotulo_coerente(bloco):
    """False when the block name says dissociation (two atoms among the products,
    e.g. N + N(2D)) while its PROCESS line describes excitation to a molecular
    state (N2( on the product side, or Excitation with no atom). An empty
    PROCESS line is unknown, not incoherent."""
    if "->" not in bloco.nome or not bloco.processo:
        return True
    produtos_nome = _produtos(re.split(r"<->|->", bloco.nome, maxsplit=1)[1])
    if len(_atomos(produtos_nome)) < 2:
        return True
    lado, _, tipo = bloco.processo.partition(",")
    lado = lado.split("->", 1)[1] if "->" in lado else lado
    molecular = re.search(r"[A-Z][a-z]?\d+\s*\(", lado) is not None
    excitacao_sem_atomo = "excitation" in tipo.lower() and not _atomos(_produtos(lado))
    return not (molecular or excitacao_sem_atomo)


ENTRADAS_CASO = ("kinet.inp", "quimica.json", "bolsigdb.dat", "transporte.json", "gas.json", "tabela_energia_tg.dat")


def _sha256(caminho):
    """SHA-256 of a file, or None when it does not exist."""
    if not caminho.is_file():
        return None
    h = hashlib.sha256()
    with open(caminho, "rb") as f:
        for bloco in iter(lambda: f.read(1 << 20), b""):
            h.update(bloco)
    return h.hexdigest()


def _importados(arquivo):
    """Dotted module names imported anywhere in a Python file (functions included)."""
    arvore = ast.parse(arquivo.read_text(), str(arquivo))
    nomes = []
    for no in ast.walk(arvore):
        if isinstance(no, ast.Import):
            nomes += [a.name for a in no.names]
        elif isinstance(no, ast.ImportFrom) and no.level == 0 and no.module:
            nomes.append(no.module)
            nomes += [f"{no.module}.{a.name}" for a in no.names]
    return nomes


def codigo_do_script(raiz, script):
    """Repository-relative paths of `script` and of every .py file of the repository it
    imports, transitively (static `import` and `from ... import` found with ast). Modules are
    searched in dbd1d, scripts, ferramentas and the importing file's folder, the folders the
    scripts put on sys.path. Sorted."""
    raiz = pathlib.Path(raiz).resolve()
    script = pathlib.Path(script).resolve()
    pastas = [raiz / "dbd1d", raiz / "scripts", raiz / "ferramentas"]
    vistos, fila = set(), [script]
    while fila:
        arq = fila.pop()
        if arq in vistos:
            continue
        vistos.add(arq)
        for nome in _importados(arq):
            partes = nome.split(".")
            for pasta in pastas + [arq.parent]:
                for cand in (pasta.joinpath(*partes).with_suffix(".py"), pasta.joinpath(*partes, "__init__.py")):
                    if cand.is_file() and raiz in cand.resolve().parents:
                        fila.append(cand.resolve())
    return sorted(str(a.relative_to(raiz)) for a in vistos)


def versao_do_pacote():
    """VERSAO of dbd1d/versao.py, read as text (this module does not import dbd1d)."""
    m = re.search(r'^VERSAO = "([^"]+)"', (RAIZ / "dbd1d" / "versao.py").read_text(), re.M)
    return m.group(1)


def registro(raiz, codigo, caso=None, extras=()):
    """Provenance record of a run: commit, SHA-256 of the code files and of the six
    input files of the case, and `sujo`, the code and case input files whose content
    is not in the commit (modified, staged or untracked). `extras` are other input
    files the run reads (paths relative to `raiz`): each is hashed in `sha256_entradas`
    under its relative path and checked for `sujo` like the case inputs. Other dirty
    files of the repository are ignored. No network access. Raises RuntimeError when git
    fails, the commit is empty, a code file or an extra is missing; a missing case input
    is recorded as None. Outside any git repository (the archive of a release) `commit` and
    `sujo` are None and `versao`, the package version, identifies the code with the hashes."""
    raiz = pathlib.Path(raiz).resolve()
    entradas = {}
    if caso is not None:
        caso = pathlib.Path(caso).resolve()
        entradas = {nome: caso / nome for nome in ENTRADAS_CASO}
    faltam = [a for a in codigo if not (raiz / a).is_file()]
    if faltam:
        raise RuntimeError("code files not found: " + ", ".join(faltam))
    faltam = [a for a in extras if not (raiz / a).is_file()]
    if faltam:
        raise RuntimeError("extra input files not found: " + ", ".join(faltam))
    rel = ([a for a in codigo] + [str(p.relative_to(raiz)) for p in entradas.values()]
           + [str(a) for a in extras])

    def git(*args):
        r = subprocess.run(["git", *args], cwd=raiz, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"git {' '.join(args[:2])} failed in {raiz}: {r.stderr.strip()}")
        return r.stdout

    hashes = dict(versao=versao_do_pacote(), sha256={a: _sha256(raiz / a) for a in codigo},
                  sha256_entradas={**{nome: _sha256(p) for nome, p in entradas.items()},
                                   **{str(a): _sha256(raiz / a) for a in extras}})
    if not any((d / ".git").exists() for d in (raiz, *raiz.parents)):
        return dict(commit=None, sujo=None, **hashes)
    commit = git("rev-parse", "HEAD").strip()
    if not commit:
        raise RuntimeError("git rev-parse HEAD returned an empty commit")
    sujo = []
    for linha in git("status", "--porcelain", "-uall", "--", *rel).splitlines():
        caminho = linha[3:].split(" -> ")[-1].strip('"')
        if caminho in rel and caminho not in sujo:
            sujo.append(caminho)
    return dict(commit=commit, sujo=sorted(sujo), **hashes)


def grava_origem(csv, script, casos, extras=()):
    """Writes CSV.origem.json beside a CSV that goes to an article: the SHA-256 of the CSV and,
    for each case of `casos` (names in casos/), the provenance record of the run (`registro`) with
    the code that `script` imports. A text that cites the CSV can then be checked against the
    current inputs and code (phase 5 of docs/auditoria/2026-11-11_cadeia/PLANO.md). Returns the
    path of the record. Without cases (a post-processor of runs), the record of the code and of
    `extras` goes under "codigo"."""
    import json
    csv = pathlib.Path(csv)
    codigo = codigo_do_script(RAIZ, script)
    reg = dict(csv=csv.name, sha256_csv=_sha256(csv),
               casos={c: registro(RAIZ, codigo, RAIZ / "casos" / c, extras=extras) for c in casos})
    if not casos:   # a post-processor of runs: the code and the files it read, without a case
        reg["codigo"] = registro(RAIZ, codigo, None, extras=extras)
    destino = csv.with_name(csv.name + ".origem.json")
    destino.write_text(json.dumps(reg, indent=1, ensure_ascii=False) + "\n")
    return destino


def exige_arvore_limpa(reg):
    """Return `reg` with `permite_sujo` added. Raise RuntimeError when `reg["sujo"]` is not
    empty, unless DBD1D_PERMITE_SUJO=1, which is then recorded in the result."""
    permite = os.environ.get("DBD1D_PERMITE_SUJO") == "1"
    if reg["sujo"] and not permite:  # None outside a repository: nothing to compare
        raise RuntimeError("files changed outside a commit, the result would not be reproducible: "
                           + ", ".join(reg["sujo"]) + ". Commit them or set DBD1D_PERMITE_SUJO=1.")
    return dict(reg, permite_sujo=permite)
