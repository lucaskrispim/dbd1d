# SPDX-License-Identifier: GPL-3.0-or-later
"""ferramentas/vv_sun.py: the N2-N2 V-V block of the mechanism of Sun et al. 2022 (supplement
PSSTac8e2csupp2.inp, as in casos/n2_amonia) with exp(+delta |v - w|), the misprint of eq. 7.38
of Capitelli et al. 2000, and with the same coefficient in both directions (ERROS.md A5, A7)."""
import math
import re
import sys

import pytest

from conftest import RAIZ, precisa_arquivo

sys.path.insert(0, str(RAIZ / "ferramentas"))
import vv_sun  # noqa: E402
from escada_vibracional import energia  # noqa: E402

LINHA = re.compile(r"^(N2(?:\(v\d+\))?) \+ (N2(?:\(v\d+\))?) => (N2(?:\(v\d+\))?) \+ (N2(?:\(v\d+\))?)\s*!\s*(kVV10_N2N2\*.*)$")


def _nivel(nome):
    m = re.fullmatch(r"N2\(v(\d+)\)", nome)
    return int(m.group(1)) if m else 0


def _avalia(expr, t):
    expr = expr.replace("kVV10_N2N2", "1.0").replace("Tgas", repr(t))
    expr = re.sub(r"(\d)d([+-]?\d)", r"\1e\2", expr)
    return eval(expr, {"exp": math.exp})


def _taxas(texto, t):
    """{(reactant levels, product levels): coefficient / kVV10} of the N2-N2 V-V lines."""
    saida = {}
    for linha in texto.splitlines():
        m = LINHA.match(linha.strip())
        if m:
            r = tuple(sorted((_nivel(m.group(1)), _nivel(m.group(2)))))
            p = tuple(sorted((_nivel(m.group(3)), _nivel(m.group(4)))))
            saida[(r, p)] = _avalia(m.group(5).split("!")[0].strip(), t)
    return saida


SINTETICO = """# V-V reactions of N2
N2(v1) + N2(v7) => N2 + N2(v8)             ! kVV10_N2N2*8.36*exp(12.1d0*(14.0d0/Tgas)**0.5)
N2(v8) + N2 => N2(v7) + N2(v1)             ! kVV10_N2N2*8.36*exp(12.1d0*(14.0d0/Tgas)**0.5)
N2(v2) + N2(v2) => N2(v1) + N2(v3)         ! kVV10_N2N2*6.11*exp(1.73d0*(14.0d0/Tgas)**0.5)
N2(v3) + N2(v1) => N2(v2) + N2(v2)         ! kVV10_N2N2*6.11*exp(1.73d0*(14.0d0/Tgas)**0.5)
N2(v8) + N => N2(v7) + N                   ! kVT10_N2N*exp(-5507.34d0/Tgas)
"""


def test_corrige_sinal_e_balanco_detalhado_no_sintetico():
    t = 500.0
    k = _taxas(vv_sun.corrige(SINTETICO), t)
    d = 12.1 * (14.0 / t) ** 0.5
    # exothermic direction (1 -> 0 gives more than 7 -> 8 takes): F-form of eq. 3.69, falling
    assert k[((1, 7), (0, 8))] == pytest.approx(8.36 * math.exp(-d) * (1.5 - 0.5 * math.exp(-d)), rel=1e-12)
    # endothermic direction by detailed balance
    de = (energia(1) - energia(0)) - (energia(8) - energia(7))
    assert de > 0
    assert k[((0, 8), (1, 7))] == pytest.approx(k[((1, 7), (0, 8))] * math.exp(-de / t), rel=1e-6)
    de2 = (energia(2) - energia(1)) - (energia(3) - energia(2))
    assert k[((1, 3), (2, 2))] == pytest.approx(k[((2, 2), (1, 3))] * math.exp(-de2 / t), rel=1e-6)


def test_outras_linhas_ficam_iguais():
    saida = vv_sun.corrige(SINTETICO).splitlines()
    assert saida[0] == "# V-V reactions of N2"
    assert saida[-1] == SINTETICO.splitlines()[-1]


def test_idempotente():
    uma = vv_sun.corrige(SINTETICO)
    assert vv_sun.corrige(uma) == uma


def test_recusa_expoente_que_nao_e_o_de_sun():
    # the tool rewrites only what it recognizes: X = 1.73 |v - w| of the supplement
    with pytest.raises(ValueError, match="exponent"):
        vv_sun.corrige("N2(v1) + N2(v7) => N2 + N2(v8)   ! kVV10_N2N2*8.36*exp(5.0d0*(14.0d0/Tgas)**0.5)\n")


# chemistry A and the cases derived from it (the minimax reviewer of wave 1 saw that only
# n2_amonia was checked)
QUIMICA_A = ["n2_amonia", "n2_amonia_a2", "n2_amonia_etapas",
             *(f"n2_amonia_superelastica/tv{t}" for t in (1000, 2000, 3000, 4500, 6000))]


def test_os_casos_da_quimica_a_estao_na_lista():
    com_bloco = {str(p.parent.relative_to(RAIZ / "casos")) for p in (RAIZ / "casos").glob("n2_amonia*/**/kinet.inp")
                 if "kVV10_N2N2*" in p.read_text()}
    assert com_bloco <= set(QUIMICA_A)


@pytest.mark.parametrize("caso", QUIMICA_A)
@pytest.mark.parametrize("t", [300.0, 500.0, 900.0])
def test_n2_amonia_satisfaz_balanco_detalhado(caso, t):
    arquivo = RAIZ / "casos" / caso / "kinet.inp"
    if not arquivo.exists():
        pytest.skip(f"{caso} not distributed")
    k = _taxas(arquivo.read_text(), t)
    assert len(k) > 50
    for (r, p), kf in k.items():
        if (p, r) not in k:
            continue
        # released energy of r -> p (Morse levels of the ladder)
        de = sum(energia(v) for v in r) - sum(energia(v) for v in p)
        if de > 0:
            assert k[(p, r)] / kf == pytest.approx(math.exp(-de / t), rel=1e-4), (r, p)
