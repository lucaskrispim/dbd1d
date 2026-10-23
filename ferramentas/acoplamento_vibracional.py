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
  superelastic inverse. No electron impact vibrational excitation from v > 0.
- The kinet.inp gets the levels in the BOLSIG block and, for each level, the
  reactions of BOLSIG+ of N2 that change the species (not the vibrational nor
  the rotational ones), with the process names of the copies.

Usage (from the scripts): banco_acoplado, kinet_acoplado, composicao.
"""
import re

from escada_vibracional import energia

EV_K = 11604.518
LIMIAR_MINIMO = 0.3          # eV
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
    if tipo in ("ELASTIC", "EFFECTIVE", "ROTATION") or not deslocar:
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


def banco_acoplado(texto, vmax, deslocar=1.0, superelastica=True, minimo=LIMIAR_MINIMO):
    """bolsigdb.dat with the copies for N2(v1..vmax) and the superelastic inverse
    of N2 -> N2(v1..v8). deslocar=False copies the cross sections of N2 as they are
    and superelastica=False keeps the single arrow (the steps of the decomposition
    of the coupling); deslocar between 0 and 1 shifts by that fraction of E_v and
    minimo replaces LIMIAR_MINIMO (tests of sensitivity)."""
    _confere(deslocar, minimo)
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


def kinet_acoplado(texto, vmax, banco_base=None, deslocar=1.0, minimo=LIMIAR_MINIMO):
    """kinet.inp with N2(v1..vmax) in the BOLSIG block and, for each level, the
    BOLSIG+ reactions of N2 that change the species (electronic excitation,
    dissociation, ionization), restricted to the processes whose shifted threshold
    stays above LIMIAR_MINIMO (known from banco_base when given, else from the
    thresholds in the comments are not known and all are kept); deslocar=False
    keeps all of them, as banco_acoplado with deslocar=False; deslocar and minimo
    as in banco_acoplado."""
    _confere(deslocar, minimo)
    linhas = texto.splitlines()
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
        if secao == "REACTIONS" and "BOLSIG N2 ->" in linha and linha.lstrip().startswith("E + N2 =>"):
            j = i + 1
            while j < len(linhas) and linhas[j].lstrip().startswith("@"):
                j += 1
            equacao, comentario = linha.split("!", 1)
            produtos = equacao.split("=>")[1].strip()
            processo = comentario.split("->", 1)[1].strip()
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


def composicao(populacoes):
    """composicao.txt from {v: fraction} (v = 0 is N2); the levels without
    population are left out."""
    if not any(f > 0.0 for f in populacoes.values()):
        raise ValueError("composicao: no level with a positive population")
    linhas = [f"{'N2' if v == 0 else f'N2(v{v})'} {f:.8g}" for v, f in sorted(populacoes.items()) if f > 0.0]
    return "\n".join(linhas + ["N2^+ traco"]) + "\n"
