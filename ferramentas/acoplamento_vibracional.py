# SPDX-License-Identifier: GPL-3.0-or-later
"""Coupling of the electron kinetics to the vibrational distribution of N2(X): a
case whose BOLSIG+ mixture has the levels N2(v1..vmax) as targets, with the
populations of the ladder (ferramentas/escada_vibracional.py).

- Each N2(v) gets a copy of the cross sections of N2 except the vibrational
  excitations: the elastic and the rotational as they are, the electronic
  excitations, dissociation and ionization with the energy axis shifted down by
  E_v, sigma_v(eps) = sigma_X(eps + E_v) (the same approximation as for the
  ionization from N2(A) in scripts/coluna_n2_etapas.py). A process whose shifted
  threshold falls below LIMIAR_MINIMO is left out (the level already lies above
  the upper state).
- The excitations N2 -> N2(v1..v8) get the double arrow: BOLSIG+ adds their
  superelastic inverse. The resonant part of 0 -> 1 of the Phelps set, N2(v1res),
  has no level of its own to come back from, so with the inverse it is summed into
  N2 <-> N2(v1) (funde_v1res), and the kinet.inp drops it from the @R of 0 -> 1.
- ev=True: electron impact vibrational excitation from the excited levels,
  N2(v) <-> N2(v + n), with sigma(v -> v + n) = sigma(0 -> n)/(1 + 0.05 v) and the
  threshold E(v + n) - E(v) (Gordiets et al. 1995, eq. (10)), for v up to
  VMAX_EV and the n of the sections of N2 in the database (v1 and v1res summed
  for n = 1), with the superelastic inverse when superelastica. Without it the
  molecules in v > 0 take no vibrational energy from the electrons.
- The kinet.inp gets the levels in the BOLSIG block and, for each level, the
  reactions of BOLSIG+ of N2 that change the species (not the vibrational nor
  the rotational ones), with the process names of the copies.

The self-consistent solution (scripts/coluna_n2_raiz.py) is a root in E/N of
the electron balance, each point with the vibrational distribution consistent
with the tables built from it: ponto_fixo (inner, at fixed E/N) and
raiz_autoconsistente (outer).

Usage (from the scripts): banco_acoplado, kinet_acoplado, composicao, ponto_fixo,
raiz_autoconsistente.
"""
import re

import numpy as np
from scipy.optimize import brentq

from escada_vibracional import energia

EV_K = 11604.518
LIMIAR_MINIMO = 0.3          # eV
VMAX_EV = 20                 # Gordiets et al. 1995: no e-V from the levels above v = 20
TIPOS = ("ELASTIC", "EFFECTIVE", "EXCITATION", "IONIZATION", "ATTACHMENT", "ROTATION")


def blocos(texto):
    """Blocks of a bolsigdb.dat: ("bloco", lines) and ("texto", [line])."""
    linhas = texto.splitlines()
    saida, i = [], 0
    while i < len(linhas):
        if linhas[i].strip() in TIPOS:
            j, tracos = i + 1, 0
            while tracos < 2:
                if j >= len(linhas):
                    alvo = linhas[i + 1].strip() if i + 1 < len(linhas) else ""
                    raise ValueError(f"block {linhas[i].strip()} {alvo} without the two dashed lines of its table")
                if linhas[j].startswith("-----"):
                    tracos += 1
                j += 1
            saida.append(("bloco", linhas[i:j]))
            i = j
        else:
            saida.append(("texto", [linhas[i]]))
            i += 1
    return saida


def limiar_deslocado(limiar, v, deslocar=1.0, minimo=LIMIAR_MINIMO):
    """Threshold (eV) from N2(v), lowered by the fraction deslocar of E_v, or None
    when it falls below minimo."""
    novo = limiar - deslocar * energia(v) / EV_K
    return novo if novo >= minimo else None


def _confere(deslocar, minimo):
    if not 0.0 <= deslocar <= 1.0:
        raise ValueError(f"deslocar is the fraction of E_v in the shift, between 0 and 1, not {deslocar}")
    if minimo < 0.0:
        raise ValueError(f"minimo, the minimum shifted threshold, cannot be negative ({minimo})")


def _vibracional(alvo_linha):
    destino = alvo_linha.split("->")[-1].strip()
    return destino.startswith("N2(v")


def _copia(bloco, v, deslocar=1.0, minimo=LIMIAR_MINIMO):
    tipo, alvo = bloco[0].strip(), bloco[1]
    nome = f"N2(v{v})"
    # rotation is copied unshifted whatever its type: in the real banks it is an
    # EXCITATION block N2 -> N2(rot), which a shift by E_v would push under the minimum
    rotacao = "rot" in alvo.split("->")[-1].lower()
    if tipo in ("ELASTIC", "EFFECTIVE", "ROTATION") or rotacao or not deslocar:
        return [bloco[0], alvo.replace("N2", nome, 1)] + [l.replace("e / N2", f"e / {nome}") for l in bloco[2:]]
    limiar = float(bloco[2].split("/")[0])
    novo = limiar_deslocado(limiar, v, deslocar, minimo)
    if novo is None:
        return None
    ev = deslocar * energia(v) / EV_K
    saida = [bloco[0], alvo.replace("N2", nome, 1), f"{novo:.6e}/threshold energy"]
    dentro = False
    for l in bloco[3:]:
        if l.startswith("-----"):
            dentro = not dentro
            saida.append(l)
            continue
        if dentro:
            e, s = map(float, l.split())
            if e >= limiar:
                saida.append(f" {e - ev:.6e}\t{s:.6e}")
            elif e == 0.0:
                saida.append(f" {0.0:.6e}\t{0.0:.6e}")
        else:
            saida.append(l.replace("e / N2", f"e / {nome}"))
    return saida


def _tabela(bloco):
    dados = []
    dentro = False
    for l in bloco[2:]:
        if l.startswith("-----"):
            dentro = not dentro
        elif dentro:
            dados.append(tuple(map(float, l.split())))
    return np.array(dados).T


def secoes_vibracionais(texto):
    """{n: (energy, cross section)} of N2 -> N2(vn) in a bolsigdb.dat, with
    N2(v1res) added to N2(v1) on the union of the two energy grids."""
    secoes = {}
    for tipo, linhas in blocos(texto):
        m = re.fullmatch(r"N2 <?-> N2\(v(\d+)(res)?\)", linhas[1].strip()) if tipo == "bloco" else None
        if not m:
            continue
        n, (e, s) = int(m[1]), _tabela(linhas)
        if n in secoes:
            e0, s0 = secoes[n]
            grade = np.union1d(e0, e)
            s = np.interp(grade, e0, s0, left=0.0, right=s0[-1]) + np.interp(grade, e, s, left=0.0, right=s[-1])
            e = grade
        secoes[n] = (e, s)
    return secoes


def funde_v1res(texto):
    """bolsigdb.dat with N2 -> N2(v1res) summed into N2 -> N2(v1) (secoes_vibracionais)
    and its block removed, so that the double arrow of v1 covers the whole 0 -> 1
    excitation; unchanged without N2(v1res)."""
    alvos = [linhas[1].strip() for tipo, linhas in blocos(texto) if tipo == "bloco"]
    if "N2 -> N2(v1res)" not in alvos:
        return texto
    if "N2 -> N2(v1)" not in alvos:
        raise ValueError("funde_v1res: N2(v1res) without N2(v1)")
    e, s = secoes_vibracionais(texto)[1]
    saida = []
    for tipo, linhas in blocos(texto):
        alvo = linhas[1].strip() if tipo == "bloco" else None
        if alvo == "N2 -> N2(v1res)":
            if saida and not saida[-1].strip():
                saida.pop()
            continue
        if alvo == "N2 -> N2(v1)":
            k = next(i for i, l in enumerate(linhas) if l.startswith("-----"))
            linhas = linhas[:k + 1] + [f" {x:.6e}\t{y:.6e}" for x, y in zip(e, s)] + [linhas[k]]
        saida.extend(linhas)
    return "\n".join(saida) + "\n"


def pares_ev(secoes, vmax, vmax_ev=VMAX_EV):
    """(v, v + n) of the e-V processes from the excited levels."""
    return [(v, v + n) for v in range(1, min(vmax_ev, vmax - 1) + 1) for n in sorted(secoes) if v + n <= vmax]


def _blocos_ev(secoes, vmax, superelastica, vmax_ev=VMAX_EV):
    saida, seta = [], "<->" if superelastica else "->"
    for v, w in pares_ev(secoes, vmax, vmax_ev):
        e, s = secoes[w - v]
        de = (energia(w) - energia(v)) / EV_K
        saida += ["", "EXCITATION", f"N2(v{v}) {seta} N2(v{w})", f"{de:.6e}/threshold energy",
                  f"SPECIES: e / N2(v{v})", f"PROCESS: E + N2(v{v}) -> E + N2(v{w}), Excitation",
                  "COLUMNS: Energy (eV) | Cross section (m2)", "-----------------------------"]
        saida += [f" {x:.6e}\t{y / (1.0 + 0.05 * v):.6e}" for x, y in zip(e, s)]
        saida.append("-----------------------------")
    return saida


def banco_acoplado(texto, vmax, deslocar=1.0, superelastica=True, minimo=LIMIAR_MINIMO, ev=False):
    """bolsigdb.dat with the copies for N2(v1..vmax) and the superelastic inverse
    of N2 -> N2(v1..v8). deslocar=False copies the cross sections of N2 as they are
    and superelastica=False keeps the single arrow (the steps of the decomposition
    of the coupling); deslocar between 0 and 1 shifts by that fraction of E_v and
    minimo replaces LIMIAR_MINIMO (tests of sensitivity); ev adds the e-V processes
    from the excited levels (module docstring)."""
    _confere(deslocar, minimo)
    if superelastica:
        texto = funde_v1res(texto)
    saida, copias = [], []
    for tipo, linhas in blocos(texto):
        if tipo == "texto":
            saida.extend(linhas)
            continue
        alvo = linhas[1].split("<->")[0].split("->")[0].strip()
        if alvo == "N2" and _vibracional(linhas[1]):
            destino = linhas[1].split("->")[-1].strip()
            if superelastica and re.fullmatch(r"N2\(v\d+\)", destino):
                linhas = [linhas[0], f"N2 <-> {destino}"] + linhas[2:]
            saida.extend(linhas)
            continue
        saida.extend(linhas)
        if alvo == "N2":
            for v in range(1, vmax + 1):
                c = _copia(linhas, v, deslocar, minimo)
                if c is not None:
                    copias.extend([""] + c)
    if ev:
        copias.extend(_blocos_ev(secoes_vibracionais(texto), vmax, superelastica))
    return "\n".join(saida + copias) + "\n"


def _processos_validos(banco_base, vmax, deslocar=1.0, minimo=LIMIAR_MINIMO):
    """{v: set of process names 'N2(vk) -> X'} that exist in the coupled database."""
    validos = {}
    for tipo, linhas in blocos(banco_base):
        if tipo == "texto" or linhas[1].split("->")[0].strip() != "N2" or _vibracional(linhas[1]):
            continue
        if linhas[0].strip() in ("ELASTIC", "EFFECTIVE", "ROTATION"):
            continue
        limiar = float(linhas[2].split("/")[0])
        destino = linhas[1].split("->")[-1].strip()
        for v in range(1, vmax + 1):
            if not deslocar or limiar_deslocado(limiar, v, deslocar, minimo) is not None:
                validos.setdefault(v, set()).add(destino)
    return validos


def kinet_acoplado(texto, vmax, banco_base=None, deslocar=1.0, minimo=LIMIAR_MINIMO, superelastica=True, ev=False):
    """kinet.inp with N2(v1..vmax) in the BOLSIG block and, for each level, the
    BOLSIG+ reactions of N2 that change the species (electronic excitation,
    dissociation, ionization), restricted to the processes whose shifted threshold
    stays above LIMIAR_MINIMO (known from banco_base when given, else from the
    thresholds in the comments are not known and all are kept); deslocar=False
    keeps all of them, as banco_acoplado with deslocar=False; deslocar and minimo
    as in banco_acoplado. superelastica (as in banco_acoplado): the inverse
    E + N2(vk) => E + N2 of each excitation N2 -> N2(vk), with the rate that
    BOLSIG+ computes for the superelastic collision, so that the energy the
    electrons gain is taken from the level. ev (as in banco_acoplado, which needs
    banco_base): E + N2(v) => E + N2(v + n) and, with superelastica, the inverse."""
    _confere(deslocar, minimo)
    if ev and banco_base is None:
        raise ValueError("ev needs banco_base, to know the vibrational sections of N2")
    linhas = texto.splitlines()
    if superelastica:
        linhas = _sem_v1res(linhas)
    saida, secao, novas = [], None, []
    validos = _processos_validos(banco_base, vmax, deslocar, minimo) if banco_base else None
    i = 0
    while i < len(linhas):
        linha = linhas[i]
        cabeca = linha.strip().upper()
        if cabeca in ("BOLSIG", "SPECIES", "REACTIONS", "ELEMENTS"):
            secao = cabeca
            saida.append(linha)
            i += 1
            if cabeca == "BOLSIG":
                bloco = []
                while linhas[i].strip().upper() != "END":
                    bloco.append(linhas[i])
                    i += 1
                presentes = " ".join(bloco).upper()
                extra = [f"N2(v{k})" for k in range(1, vmax + 1) if f"N2(V{k})" not in presentes.split()]
                todas = [t for l in bloco for t in l.split()] + extra
                # lines of 8 names: the preprocessor of ZDPlasKin cuts long lines
                saida.extend("  ".join(todas[k:k + 8]) for k in range(0, len(todas), 8))
            continue
        if cabeca == "END" and secao == "REACTIONS":
            saida.extend(novas)
            if ev:
                for v, w in pares_ev(secoes_vibracionais(banco_base), vmax):
                    saida.append(f"E + N2(v{v}) => E + N2(v{w}) ! BOLSIG N2(v{v}) -> N2(v{w})")
                    if superelastica:
                        saida.append(f"E + N2(v{w}) => E + N2(v{v}) ! BOLSIG N2(v{w}) -> N2(v{v})")
        if secao == "REACTIONS" and "BOLSIG N2 ->" in linha and linha.lstrip().startswith("E + N2 =>"):
            j = i + 1
            while j < len(linhas) and linhas[j].lstrip().startswith("@"):
                j += 1
            equacao, comentario = linha.split("!", 1)
            produtos = equacao.split("=>")[1].strip()
            processo = comentario.split("->", 1)[1].strip()
            if superelastica and produtos.startswith("E + N2(v"):
                lista = linhas[i + 1].split("=", 1)[1].split() if processo == "@R" else [processo]
                # the same levels that get the double arrow in banco_acoplado
                for nivel in (p for p in lista if re.fullmatch(r"N2\(v\d+\)", p)):
                    novas.append(f"E + {nivel} => E + N2 ! BOLSIG {nivel} -> N2")
            if produtos not in ("E + N2",) and not produtos.startswith("E + N2(v"):
                lista = linhas[i + 1].split("=", 1)[1].split() if processo == "@R" else [processo]
                for v in range(1, vmax + 1):
                    ok = [p for p in lista if validos is None or p in validos.get(v, set())]
                    if not ok:
                        continue
                    r = f"E + N2(v{v}) => {produtos} ! BOLSIG N2(v{v}) -> " + ("@R" if processo == "@R" else ok[0])
                    novas.append(r)
                    if processo == "@R":
                        novas.append("    @R = " + "  ".join(ok))
            saida.extend(linhas[i:j])
            i = j
            continue
        saida.append(linha)
        i += 1
    return "\n".join(saida) + "\n"


def _sem_v1res(linhas):
    """kinet.inp lines with N2(v1res) out of the @R lists of BOLSIG N2 -> @R (it is
    summed into N2(v1) by funde_v1res); an @R left with one process is written
    directly."""
    saida, i = [], 0
    while i < len(linhas):
        l = linhas[i]
        if ("BOLSIG N2 -> @R" in l and i + 1 < len(linhas) and linhas[i + 1].lstrip().startswith("@R")
                and "N2(v1res)" in linhas[i + 1].split()):
            cabeca, nomes = linhas[i + 1].split("=", 1)
            nomes = [n for n in nomes.split() if n != "N2(v1res)"]
            if len(nomes) == 1:
                saida.append(l.replace("BOLSIG N2 -> @R", f"BOLSIG N2 -> {nomes[0]}"))
            else:
                saida += [l, f"{cabeca}= " + "  ".join(nomes)]
            i += 2
            continue
        saida.append(l)
        i += 1
    return saida


def composicao(populacoes):
    """composicao.txt from {v: fraction} (v = 0 is N2); the levels without
    population are left out."""
    if not any(f > 0.0 for f in populacoes.values()):
        raise ValueError("composicao: no level with a positive population")
    linhas = [f"{'N2' if v == 0 else f'N2(v{v})'} {f:.8g}" for v, f in sorted(populacoes.items()) if f > 0.0]
    return "\n".join(linhas + ["N2^+ traco"]) + "\n"


def ponto_fixo(mapa, p0, tol=1e-3, max_it=12, piso=1e-14):
    """Iterates p -> mapa(p) until the largest change of ln p, over the entries
    above piso in both iterates, is below tol. Returns (p, iterations, changes)."""
    p, mudancas = np.asarray(p0, dtype=float), []
    for k in range(1, max_it + 1):
        novo = np.asarray(mapa(p), dtype=float)
        vivos = (novo > piso) & (p > piso)
        mudancas.append(float(np.max(np.abs(np.log(novo[vivos] / p[vivos])))) if vivos.any() else 0.0)
        p = novo
        if mudancas[-1] < tol:
            return p, k, mudancas
    raise RuntimeError(f"fixed point did not converge in {max_it} iterations: last change {mudancas[-1]:.5g}")


def raiz_autoconsistente(avalia, x0, x_min, xtol=0.05, passo_min=0.5, max_aval=40):
    """Root in x (E/N, Td) of the electron balance r(x), with avalia(x) -> (r, phi):
    r the relative balance (positive: production above the loss, x too high) and
    phi the E/N that the balance gives with the state of x held (the step of the
    fixed-point iteration), or None. From x0 the steps follow phi at first and then
    the secant of r, aimed 10% beyond its zero so as to bracket it, within a factor
    passo_min of x and not below x_min; once r changes sign, Brent's method.
    Each x is evaluated once. Returns (x, [(x, r, phi), ...])."""
    cache, hist = {}, []

    def f(x):
        if x not in cache:
            cache[x] = avalia(x)
            hist.append((x,) + tuple(cache[x]))
        return cache[x]

    x, anterior = x0, None
    r, phi = f(x)
    while r != 0.0:
        sobe = r < 0.0
        alvo = None
        if anterior is not None and (abs(r) < abs(anterior[1])):
            xs = x - r * (x - anterior[0]) / (r - anterior[1])
            alvo = xs + 0.1 * (xs - x)
        if alvo is None and phi is not None and np.isfinite(phi) and (phi > x) == sobe and phi != x:
            alvo = phi
        if alvo is None:
            alvo = x / passo_min if sobe else x * passo_min
        xn = min(max(alvo, passo_min * x, x_min), x / passo_min)
        if xn == x:
            raise ValueError(f"no root of the electron balance between {x_min:g} and {x0:g} Td: "
                             f"{'negative' if sobe else 'positive'} at both ends")
        rn, phin = f(xn)
        if rn == 0.0:
            return xn, hist
        if (rn > 0.0) != (r > 0.0):
            raiz = brentq(lambda z: f(z)[0], min(x, xn), max(x, xn), xtol=xtol)
            return raiz, hist
        if len(hist) >= max_aval:
            raise RuntimeError(f"no change of sign of the electron balance in {max_aval} evaluations")
        anterior, (x, r, phi) = (x, r), (xn, rn, phin)
    return x, hist
