# SPDX-License-Identifier: GPL-3.0-or-later
"""Correction of the N2-N2 V-V block of the mechanism of Sun et al. 2022 (supplement
PSSTac8e2csupp2.inp), which the cases built on it carry (casos/n2_amonia and the others;
docs/auditoria/2026-11-10/ERROS.md A5 and A7).

Each line N2(a) + N2(b) => N2(a - 1) + N2(b + 1) of the supplement has the coefficient
kVV10_N2N2 * Z_a Z_(b+1) * exp(+X (14/Tgas)^0.5), with X = 1.73 |v - w| (v = a - 1, w = b, the
notation of Capitelli et al. 2000, eq. 7.35), and the same coefficient in both directions:

- the sign follows eq. 7.38 as printed (p. 114); the full form of the same page (eq. 7.36, with
  F(y) of eq. 7.34) and eq. 3.69 (p. 37) fall with the energy mismatch:
  exp(-X s) (1.5 - 0.5 exp(-X s)), s = (14/Tgas)^0.5 (A5);
- the reverse of a process has the coefficient of the direct one, without the detailed balance
  factor exp(-Delta E/Tgas), Delta E = 2 omega_e x_e |a - 1 - b| (A7).

Here the exothermic direction keeps kVV10 and the Z factors of the supplement with the falling
form, and the endothermic direction is that coefficient times exp(-Delta E/Tgas), with the Morse
levels of escada_vibracional.py. Lines already corrected, and all the other lines, are kept.

Usage: python3 ferramentas/vv_sun.py FILE [FILE ...]   (rewrites each kinet.inp in place)
"""
import math
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from escada_vibracional import energia  # noqa: E402

DELTA_SUN = 1.73          # X / |v - w| of the supplement (6.47/sqrt(T) = 1.73 sqrt(14/T))
_N2 = r"N2(?:\(v\d+\))?"
LINHA = re.compile(rf"^(?P<r1>{_N2}) \+ (?P<r2>{_N2}) => (?P<p1>{_N2}) \+ (?P<p2>{_N2})(?P<esp>\s*!\s*)"
                   r"kVV10_N2N2\*(?P<z>[0-9.]+)\*exp\((?P<x>[0-9.]+)d0\*\(14\.0d0/Tgas\)\*\*0\.5\)\s*$")


def _nivel(nome):
    m = re.fullmatch(r"N2\(v(\d+)\)", nome)
    return int(m.group(1)) if m else 0


def _papeis(r, p):
    """(a, b) of N2(a) + N2(b) => N2(a - 1) + N2(b + 1), or None when the line is not that."""
    for a, b in (r, r[::-1]):
        if sorted((a - 1, b + 1)) == sorted(p):
            return a, b
    return None


def _d(x):
    return f"{x:.6f}d0"


def corrige_linha(linha):
    m = LINHA.match(linha)
    if not m:
        return linha
    r = (_nivel(m["r1"]), _nivel(m["r2"]))
    p = (_nivel(m["p1"]), _nivel(m["p2"]))
    papeis = _papeis(r, p)
    if papeis is None:
        return linha
    a, b = papeis
    x = float(m["x"])
    n = abs(a - 1 - b)
    if abs(x - DELTA_SUN * n) > 0.01 * max(n, 1):
        raise ValueError(f"exponent {x} of '{linha.strip()}' is not {DELTA_SUN} |v - w| = {DELTA_SUN * n:.2f}")
    s = f"{m['x']}d0*(14.0d0/Tgas)**0.5"
    k = f"kVV10_N2N2*{m['z']}*exp(-{s})*(1.5d0-0.5d0*exp(-{s}))"
    # energy released by the process: quantum given by a minus quantum taken by b
    liberada = (energia(a) - energia(a - 1)) - (energia(b + 1) - energia(b))
    if liberada < 0:
        k += f"*exp(-{_d(-liberada)}/Tgas)"
    cabeca = linha[:m.start("esp")]
    return f"{cabeca}{m['esp']}{k}"


def corrige(texto):
    linhas = [corrige_linha(l) for l in texto.splitlines()]
    return "\n".join(linhas) + ("\n" if texto.endswith("\n") else "")


def main(arquivos):
    for nome in arquivos:
        p = pathlib.Path(nome)
        antes = p.read_text()
        depois = corrige(antes)
        if depois != antes:
            p.write_text(depois)
        print(f"{p}: {'corrected' if depois != antes else 'already corrected'}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1:])
