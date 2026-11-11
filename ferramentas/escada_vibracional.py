# SPDX-License-Identifier: GPL-3.0-or-later
"""Vibrational ladder of N2(X) up to v = vmax for a kinet.inp (ZDPlasKin format),
with the rate coefficients of Capitelli, Ferreira, Gordiets and Osipov 2000,
Plasma Kinetics in Atmospheric Gases, chapter 7:

- levels: Morse, E_v = (we v - wexe v (v + 1)) hc/k, with we = 2358.57 and
  wexe = 14.324 cm-1;
- V-T with N2, N2(v) + N2 -> N2(v - 1) + N2: k = k10 Z_v exp((v - 1) delta_VT)
  (eqs. 7.4 to 7.8, table 7.1; delta_VT of gamma_0 >= 20, which holds below
  about 1500 K), with the anharmonic defect Delta E = omega_e x_e of eq. 3.67
  (p. 36), and the inverse by detailed balance;
- V-V, N2(v) + N2(w - 1) -> N2(v - 1) + N2(w) for v < w (the exothermic
  direction): k = k01 G_S(v, w) g(v, w) (eqs. 7.26a, 7.40, 7.41, L = 0.2 A), with
  G_S(v, w) = Z_v Z_w F(y)/F(y_10^01) of eq. 7.36 and F of eq. 7.34 (both
  branches, y <= 20 and y > 20); y_10^01 = 0 and F(0) = 1. By eq. 7.37 with the
  Morse levels, E_{v,v-1} - E_{w,w-1} = 2 omega_e x_e (w - v), so 2y/3 =
  delta_VV |v - w| with delta_VV of eq. 7.39 (0.427 is 0.32 x 2 x 2/3 rounded);
  below y = 20, F(y) = [1.5 - 0.5 exp(-delta_VV |v - w|)] exp(-delta_VV |v - w|),
  the simplified exp(-delta_VV |v - w|) of eq. 7.38 times 1.5 - 0.5 exp(...).
  Eq. 7.38 is printed with exp(+delta_VV |v - w|); the exothermic rate falls with
  the energy mismatch, as F(y) does. The inverse comes by detailed balance;
- V-T with N, multi-quantum, Delta v = 1 to 5 (eqs. 7.13 and 7.14, k0 =
  4e-10 (T/300)^0.5, E_a = 7280 K, beta = 0.065), without the inverse (as in
  Levaton et al. 2020);
- dissociation from the top of the ladder: the V-T and V-V transitions out of
  vmax lead to N + N (as V7 of Levaton et al. 2020), without the inverse.
The electron impact excitation (BOLSIG+, v = 1 to 8 from v = 0) of the base
chemistry stays; its V-T and V-V among N2(v) are replaced.

Usage: python3 ferramentas/escada_vibracional.py ORIGIN/kinet.inp DESTINATION/kinet.inp [VMAX]
"""
import math
import re
import sys
from dataclasses import dataclass

WE, WEXE = 2358.57, 14.324          # cm-1, N2(X)
HC_K = 1.438776877                  # cm K
XE = WEXE / WE
DE = WEXE * HC_K                    # K: anharmonic constant Delta E = omega_e x_e, eq. 3.67 of Capitelli 2000
MU, L = 14.0, 0.2                   # reduced mass (u) of N2-N2 and L (A)
PREFIXO = "escv_"


def energia(v):
    """Energy (K) of N2(X, v) above v = 0."""
    return (WE * v - WEXE * v * (v + 1)) * HC_K


def _z(v):
    return v * (1.0 - XE) / (1.0 - v * XE)


def _k10_vt(t):
    e10 = energia(1)
    return 7.8e-12 * t * math.exp(-218.0 / t ** (1 / 3) + 690.0 / t) / (1.0 - math.exp(-e10 / t))


def _delta_vt(t):
    return 1.87 * DE * L ** (2 / 3) * MU ** (1 / 3) / (energia(1) ** (1 / 3) * t ** (1 / 3))


def _delta_vv(t):
    return 0.427 * DE * L * math.sqrt(MU / t)


def _y(n, t):
    """y of eq. 7.37 for the mismatch |v - w| = n: 2y/3 = n delta_VV."""
    return 1.5 * n * _delta_vv(t)


# F(y) of eq. 7.34 above y = 20: F = F_ALTO y^(7/3) exp(-3 y^(2/3))
F_ALTO = 8.0 * math.sqrt(math.pi / 3.0)


def _f(y):
    """F(y)/F(0) of eqs. 7.34 and 7.36 (F(0) = 1)."""
    if y <= 20.0:
        return 0.5 * (3.0 - math.exp(-2.0 * y / 3.0)) * math.exp(-2.0 * y / 3.0)
    return F_ALTO * y ** (7.0 / 3.0) * math.exp(-3.0 * y ** (2.0 / 3.0))


def _g(v, w):
    m = max(v, w)
    return 1.0 - 0.04 * m if m < 10 else 0.645 + 0.617 * ((m - 10) / 30.0) ** 3


def k_vt(v, t):
    """N2(v) + N2 -> N2(v - 1) + N2 (cm3/s)."""
    return _k10_vt(t) * _z(v) * math.exp((v - 1) * _delta_vt(t))


def k_vv(v, w, t):
    """N2(v) + N2(w - 1) -> N2(v - 1) + N2(w), v < w (cm3/s)."""
    if not v < w:
        raise ValueError("k_vv gives the exothermic direction, v < w")
    return 0.9e-14 * (t / 300.0) ** 1.5 * _z(v) * _z(w) * _f(_y(abs(v - w), t)) * _g(v, w)


def k_vt_n(v, dv, t):
    """N2(v) + N -> N2(v - dv) + N (cm3/s)."""
    if dv < 1:
        raise ValueError(f"Delta v must be at least 1, not {dv}")
    if dv > min(v, 5):
        return 0.0
    kv = 4e-10 * (t / 300.0) ** 0.5
    if 0.065 * energia(v) <= 7280.0:
        kv *= math.exp(-7280.0 / t + 0.065 * energia(v) / t)
    return kv / min(v, 5)


# ------------------------------------------------------------------ kinet.inp

def _d(x):
    """Double precision literal for the chemistry reader (a real without D is single)."""
    return f"{x:.12e}".replace("e", "d")


def _nome(v):
    return "N2" if v == 0 else f"N2(v{v})"


@dataclass
class Reacao:
    tipo: str
    reagentes: list
    produtos: list
    taxa: str

    def linha(self):
        return f"{' + '.join(self.reagentes)} => {' + '.join(self.produtos)}".ljust(44) + f" ! {self.taxa}"


def variaveis(vmax=45):
    """$ lines with the temperature dependent factors: the V-T and V-V constants and
    F(y) of eq. 7.34 for each mismatch |v - w| = 1 to vmax (escv_f1 ...), with y of
    _y; the branch y > 20 is a one-line IF."""
    p = PREFIXO
    f = [f"{p}f{n}" for n in range(1, vmax + 1)]
    linhas_f = []
    for n in range(1, vmax + 1):
        linhas_f += [
            f"$ {p}y = {_d(1.5 * n)}*{p}dvv",
            f"$ {p}f{n} = 0.5d0*(3.0d0-exp(-2.0d0*{p}y/3.0d0))*exp(-2.0d0*{p}y/3.0d0)",
            f"$ if ({p}y > 20.0d0) {p}f{n} = {_d(F_ALTO)}*{p}y**(7.0d0/3.0d0)*exp(-3.0d0*{p}y**(2.0d0/3.0d0))",
        ]
    return [
        f"$ double precision :: {p}k10, {p}dvt, {p}k01, {p}dvv, {p}kn, {p}y",
        *(f"$ double precision :: {', '.join(f[k:k + 8])}" for k in range(0, len(f), 8)),
        f"$ {p}k10 = 7.8d-12*Tgas*exp(-218.0d0/Tgas**(1.0d0/3.0d0)+690.0d0/Tgas)/(1.0d0-exp(-{_d(energia(1))}/Tgas))",
        f"$ {p}dvt = {_d(1.87 * DE * L ** (2 / 3) * MU ** (1 / 3) / energia(1) ** (1 / 3))}/Tgas**(1.0d0/3.0d0)",
        f"$ {p}k01 = 0.9d-14*(Tgas/300.0d0)**1.5d0",
        f"$ {p}dvv = {_d(0.427 * DE * L * math.sqrt(MU))}/sqrt(Tgas)",
        f"$ {p}kn = 4.0d-10*(Tgas/300.0d0)**0.5d0",
        *linhas_f,
    ]


def reacoes(vmax):
    if vmax < 1:
        raise ValueError(f"vmax must be at least 1, not {vmax}")
    p = PREFIXO
    lista = []
    for v in range(1, vmax + 1):
        k = f"{p}k10*{_d(_z(v))}*exp({_d(v - 1)}*{p}dvt)"
        lista.append(Reacao("VT", [_nome(v), "N2"], [_nome(v - 1), "N2"], k))
        lista.append(Reacao("VT", [_nome(v - 1), "N2"], [_nome(v), "N2"],
                            f"{k}*exp(-{_d(energia(v) - energia(v - 1))}/Tgas)"))
    for v in range(1, vmax + 1):
        for w in range(v + 1, vmax + 1):
            k = f"{p}k01*{_d(_z(v) * _z(w) * _g(v, w))}*{p}f{abs(v - w)}"
            q = (energia(v) - energia(v - 1)) - (energia(w) - energia(w - 1))
            lista.append(Reacao("VV", [_nome(v), _nome(w - 1)], [_nome(v - 1), _nome(w)], k))
            lista.append(Reacao("VV", [_nome(v - 1), _nome(w)], [_nome(v), _nome(w - 1)], f"{k}*exp(-{_d(q)}/Tgas)"))
    for v in range(1, vmax + 1):
        for dv in range(1, min(v, 5) + 1):
            base = f"{p}kn/{_d(min(v, 5))}"
            if 0.065 * energia(v) <= 7280.0:
                base += f"*exp(-7280.0d0/Tgas+{_d(0.065 * energia(v))}/Tgas)"
            lista.append(Reacao("VTN", [_nome(v), "N"], [_nome(v - dv), "N"], base))
    topo = vmax + 1
    q_topo = energia(topo) - energia(vmax)
    lista.append(Reacao("DIS", [_nome(vmax), "N2"], ["N", "N", "N2"],
                        f"{p}k10*{_d(_z(topo))}*exp({_d(topo - 1)}*{p}dvt)*exp(-{_d(q_topo)}/Tgas)"))
    for v in range(1, vmax + 1):
        lista.append(Reacao("DIS", [_nome(v), _nome(vmax)], [_nome(v - 1), "N", "N"],
                            f"{p}k01*{_d(_z(v) * _z(topo) * _g(v, topo))}*{p}f{abs(v - topo)}"))
    return lista


def _especies_da_reacao(linha):
    equacao = linha.split("!")[0]
    return [t.strip() for t in re.split(r"\s\+\s|=>", equacao) if t.strip()]


def substitui_vibracao(texto, vmax=45):
    """kinet.inp with the ladder N2(v1..vmax) in place of the V-T and V-V of the base:
    the reactions whose species are only N2, N2(v) and N, with some N2(v), are
    dropped (the electron impact excitation and N + N + M stay); the species
    N2(v9..vmax), the $ variables and the reactions of the ladder are added."""
    if vmax < 1:
        raise ValueError(f"vmax must be at least 1, not {vmax}")
    vib = {f"N2(V{k})" for k in range(1, 100)}
    linhas = texto.splitlines()
    saida, secao = [], None
    i_especies = i_reacoes = None
    i = 0
    while i < len(linhas):
        linha = linhas[i]
        cabeca = linha.strip().upper()
        if cabeca in ("SPECIES", "REACTIONS", "BOLSIG", "ELEMENTS"):
            secao = cabeca
        elif cabeca == "END":
            if secao == "SPECIES":
                i_especies = len(saida)
            if secao == "REACTIONS":
                i_reacoes = len(saida)
            secao = None
        elif secao == "REACTIONS" and "=>" in linha and not linha.lstrip().startswith(("#", "$")):
            j = i + 1
            while j < len(linhas) and linhas[j].lstrip().startswith("@"):
                j += 1
            esp = {s.upper() for s in _especies_da_reacao(linha) if not s.startswith("@")}
            for b in linhas[i + 1:j]:            # the species of the @ lists (third bodies, V-T in lists)
                nome, valores = b.split("=", 1)
                if nome.strip() != "@R":
                    esp |= {v.upper() for v in valores.split()}
            if "BOLSIG" not in linha.upper() and esp & vib and esp <= vib | {"N2", "N"}:
                i = j
                continue
            saida.extend(linhas[i:j])
            i = j
            continue
        saida.append(linha)
        i += 1
    if i_especies is None or i_reacoes is None:
        raise ValueError("kinet.inp without SPECIES or REACTIONS")
    novas_especies = [f"N2(v{k})" for k in range(1, vmax + 1) if f"N2(V{k})" not in texto.upper()]
    bloco = ["", f"# vibrational ladder of N2(X) up to v = {vmax}: ferramentas/escada_vibracional.py",
             *variaveis(vmax), *(r.linha() for r in reacoes(vmax)), ""]
    saida = saida[:i_reacoes] + bloco + saida[i_reacoes:]
    for k in range(0, len(novas_especies), 8):
        saida.insert(i_especies, "  ".join(novas_especies[k:k + 8]))
        i_especies += 1
    return "\n".join(saida) + "\n"


def main():
    if len(sys.argv) not in (3, 4):
        sys.exit("Usage: python3 ferramentas/escada_vibracional.py ORIGIN/kinet.inp DESTINATION/kinet.inp [VMAX]")
    origem, destino = sys.argv[1:3]
    vmax = int(sys.argv[3]) if len(sys.argv) > 3 else 45
    with open(destino, "w") as f:
        f.write(substitui_vibracao(open(origem).read(), vmax))


if __name__ == "__main__":
    main()
