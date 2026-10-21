# SPDX-License-Identifier: GPL-3.0-or-later
"""Geração automática contra o próprio ZDPlasKin: para mecanismos conhecidos e
modificações deles, a ferramenta (ferramentas/caso.py) gera o zdplaskin_m.F90
pelo pré-processador; a sonda ferramentas/sonda_zdplaskin.f90, compilada contra
esse arquivo, deixa o ZDPlasKin e o BOLSIG+ calcularem as taxas das reações e as
fontes das espécies em vários estados; o protótipo, lendo o mesmo arquivo e com
as mesmas entradas (T_g, E/N, T_e, densidades e taxas do BOLSIG+), tem de
reproduzir cada taxa e cada fonte. Nenhum valor esperado é escrito à mão:
quem diz o certo é o ZDPlasKin."""
import shutil
import subprocess
import sys

import numpy as np
import pytest

from conftest import RAIZ

sys.path.insert(0, str(RAIZ / "ferramentas"))
import caso  # noqa: E402
from quimica_zdp import Quimica  # noqa: E402

try:
    TERCEIROS = caso.terceiros()
except caso.ErroCaso:
    TERCEIROS = None
pytestmark = [pytest.mark.lento,
              pytest.mark.skipif(TERCEIROS is None, reason="ZDPlasKin e BOLSIG+ ausentes (ZDPLASKIN_DIR)")]
N2 = RAIZ / "casos" / "n2"
AMONIA = RAIZ / "casos" / "amonia_sun2022"
OPCOES = ["-O2", "-ffp-contract=off", "-w"]


@pytest.fixture(scope="module")
def dvode(tmp_path_factory):
    """dvode_f90_m compilado uma vez (o .o e o .mod)."""
    d = tmp_path_factory.mktemp("dvode")
    subprocess.run(["gfortran", *OPCOES, "-J", str(d), "-c", str(TERCEIROS["dvode"]), "-o", str(d / "dvode.o")],
                   check=True, capture_output=True)
    return d


def _prepara(destino, origem, trocas=(), acrescimos=()):
    """Pasta de caso com o kinet.inp de origem modificado: trocas (velho, novo)
    que precisam casar uma vez, e linhas acrescentadas antes do END final."""
    destino.mkdir()
    for nome in ("bolsigdb.dat", "composicao.txt"):
        shutil.copy(origem / nome, destino / nome)
    kinet = (origem / "kinet.inp").read_text()
    for velho, novo in trocas:
        assert kinet.count(velho) == 1, velho
        kinet = kinet.replace(velho, novo)
    if acrescimos:
        i = kinet.rstrip().rfind("END")
        kinet = kinet[:i] + "\n".join(acrescimos) + "\n" + kinet[i:]
    (destino / "kinet.inp").write_text(kinet)
    caso.gera_zdplaskin(destino)
    return destino


def _estados(q, pasta):
    """Três estados: o gás da composicao.txt a ~2,4e19 cm-3 e as demais
    espécies de 1e9 a 1e13, em T_g e E/N diferentes."""
    fundo = {}
    for linha in (pasta / "composicao.txt").read_text().splitlines():
        p = linha.split()
        if p and not p[0].startswith("#") and len(p) > 1 and p[1] != "traco":
            fundo[p[0].upper()] = float(p[1])
    soma = sum(fundo.values())
    estados = []
    for k, (tg, e_n) in enumerate(((300.0, 30.0), (450.0, 150.0), (700.0, 600.0))):
        d = np.array([fundo[s.upper()] / soma * 2.4e19 if s.upper() in fundo else 10.0 ** (9 + (i + k) % 5)
                      for i, s in enumerate(q.especies)])
        estados.append((tg, e_n, d))
    return estados


def _sonda(pasta, dvode, estados):
    """Compila e roda a sonda na pasta; devolve as linhas de sonda.txt."""
    exe = pasta / "sonda.x"
    subprocess.run(["gfortran", *OPCOES, "-I", str(dvode), "-J", str(pasta), str(dvode / "dvode.o"),
                    str(pasta / "zdplaskin_m.F90"), str(RAIZ / "ferramentas" / "sonda_zdplaskin.f90"),
                    "-o", str(exe), "-L", str(TERCEIROS["bolsig"].parent), f"-l:{TERCEIROS['bolsig'].name}",
                    f"-Wl,-rpath,{TERCEIROS['bolsig'].parent}"],
                   check=True, capture_output=True)
    linhas = [f"{len(estados)}"] + [" ".join(repr(float(x)) for x in (tg, e_n, *d)) for tg, e_n, d in estados]
    (pasta / "estados.txt").write_text("\n".join(linhas) + "\n")
    subprocess.run([str(exe)], cwd=pasta, stdin=subprocess.DEVNULL, check=True, capture_output=True)
    return np.loadtxt(pasta / "sonda.txt", ndmin=2)


def _confere(pasta, dvode):
    q = Quimica(pasta / "zdplaskin_m.F90")
    estados = _estados(q, pasta)
    saida = _sonda(pasta, dvode, estados)
    nk = sum(c is not None for c in q.coluna_bolsig) and max(c for c in q.coluna_bolsig if c is not None) + 1
    nr, ns = len(q.reacoes), len(q.especies)
    assert saida.shape[1] == 4 + nk + nr + ns
    for (tg, e_n, d), linha in zip(estados, saida):
        te, neutros = linha[2], linha[3]
        kb = linha[4:4 + nk][None, :]
        taxas_f = linha[4 + nk:4 + nk + nr]
        fontes_f = linha[4 + nk + nr:]
        k = q.coeficientes(kb, np.array([tg]), np.array([te]), np.array([e_n]),
                           np.array([neutros]) if q.usa_any_neutral else None)
        taxas_p = q.taxas(d[:, None], k)[:, 0]
        np.testing.assert_allclose(taxas_p, taxas_f, rtol=1e-11, atol=0.0, err_msg=f"{pasta.name}: taxas")
        escala = np.abs(q.estequiometria) @ np.abs(taxas_f)
        assert np.all(np.abs(q.estequiometria @ taxas_p - fontes_f) <= 1e-11 * escala + 1e-300), \
            f"{pasta.name}: fontes"
    return q


def test_n2_como_esta(tmp_path, dvode):
    _confere(_prepara(tmp_path / "n2", N2), dvode)


def test_amonia_de_sun_como_esta(tmp_path, dvode):
    # 38 espécies, 492 reações, variáveis locais, ANY_NEUTRAL e @M expandidos
    q = _confere(_prepara(tmp_path / "amonia", AMONIA), dvode)
    assert len(q.reacoes) == 492


@pytest.mark.parametrize("nome, trocas, acrescimos", [
    ("coeficiente_trocado",
     [("N^+  + N  + N2 => N2^+ + N2        !   1.0d-29",
       "N^+  + N  + N2 => N2^+ + N2        !   3.7d-30 * (300.0d0/Tgas)**1.5d0")], []),
    ("especie_nova",
     [("SPECIES\ne N N2 N2(A)", "SPECIES\ne N N2 N2(W) N2(A)")],
     ["N2(A) + N2(A) => N2(W) + N2        !   1.0d-11 * exp(-150.0d0/Tgas)",
      "N2(W) => N2(B)                     !   2.0d+04",
      "N2(W) + N => N2 + N                !   4.0d-12 * sqrt(Tgas/300.0d0)"]),
    ("variavel_com_if",
     [("$ Teff = Tgas + 0.12d0 * EN**2",
       "$ Teff = Tgas + 0.12d0 * EN**2\n$ double precision :: Fator\n$ if (Tgas .gt. 400.0d0 .and. EN < 300.0d0) then\n"
       "$   Fator = 2.0d0\n$ else if (Tgas > 600.0d0) then\n$   Fator = 3.0d0\n$ else\n$   Fator = 1.0d0\n$ end if"),
      ("N2(A)  + N     => N2 + N           !   2.0d-12",
       "N2(A)  + N     => N2 + N           !   2.0d-12 * Fator")], []),
    ("divisao_inteira",
     [("N3^+ + N       => N2^+ + N2        !   6.6d-11",
       "N3^+ + N       => N2^+ + N2        !   6.6d-11 * (Tgas/300.0d0)**(3/2) + 1.0d-12*(7/2)")], []),
    ("reacao_retirada",
     [("N2(B)  + N2    => N2 + N2          !   2.0d-12\n", "")], []),
    # como no kinet.inp de Sun: ANY_NEUTRAL só na taxa; o pré-processador
    # declara a variável e chama ZDPlasKin_get_density_total
    ("any_neutral",
     [("N2(a') + N2    => N2 + N2(B)       !   1.9d-13",
       "N2(a') + N2    => N2 + N2(B)       !   1.9d-13 * ANY_NEUTRAL / 2.5d19")], []),
])
def test_n2_modificado(tmp_path, dvode, nome, trocas, acrescimos):
    _confere(_prepara(tmp_path / nome, N2, trocas, acrescimos), dvode)


def test_mistura_nova_ar_n2(tmp_path, dvode):
    # mistura que não existia: os mecanismos do argônio e do N2 juntos, com
    # reações cruzadas (casos/ar_n2/monta.py), Ar:N2 90:10
    q = _confere(_prepara(tmp_path / "ar_n2", RAIZ / "casos" / "ar_n2"), dvode)
    assert {"AR^+", "N2^+", "AR*", "N2(C)"} <= {e.upper() for e in q.especies}


def test_ar_seco_com_argonio(tmp_path, dvode):
    # o mecanismo N2-O2 do ZDPlasKin (íons negativos, NxOy, blocos @B/@R,
    # DENSITY no BOLSIG+) com o argônio e reações cruzadas (casos/ar_n2_o2/monta.py)
    q = _confere(_prepara(tmp_path / "ar_n2_o2", RAIZ / "casos" / "ar_n2_o2"), dvode)
    assert len(q.reacoes) == 667 and "O2^+N2" in {e.upper() for e in q.especies}
