# SPDX-License-Identifier: GPL-3.0-or-later
"""Rebuilding the N2 chemistries of the positive column (ferramentas/casos_coluna.py)
from the cases they come from: the kinet.inp and bolsigdb.dat written are those of
the research cases, byte for byte."""
import subprocess
import sys

import pytest

from conftest import RAIZ, precisa_arquivo, precisa_terceiros

sys.path.insert(0, str(RAIZ / "ferramentas"))
import casos_coluna as cc  # noqa: E402

FONTES = ("casos/xu2025/kinet.inp", "casos/xu2025/bolsigdb.dat", "casos/ar_n2_o2/kinet.inp",
          "casos/ar_n2_o2/bolsigdb.dat", "casos/n2/kinet.inp", "casos/n2/bolsigdb.dat")


def test_receitas_em_ordem_de_dependência():
    nomes = list(cc.RECEITAS)
    assert nomes == ["n2_exemplo", "n2_amonia", "n2_tese", "n2_amonia_v45", "n2_tese_v45", "n2_amonia_a2",
                     "n2_tese_a2"]
    for nome, (origem, _) in cc.RECEITAS.items():
        if origem in cc.RECEITAS:
            assert nomes.index(origem) < nomes.index(nome)


@precisa_arquivo(*FONTES, *(f"casos/{n}/kinet.inp" for n in cc.RECEITAS))
@pytest.mark.parametrize("nome", list(cc.RECEITAS))
def test_receita_reproduz_o_caso(tmp_path, nome):
    kinet, banco = cc.monta(nome, RAIZ / "casos")
    assert kinet == (RAIZ / "casos" / nome / "kinet.inp").read_text()
    assert banco == (RAIZ / "casos" / nome / "bolsigdb.dat").read_text()


def test_caso_desconhecido_é_recusado():
    with pytest.raises(ValueError, match="^unknown case n2_xyz: n2_exemplo, n2_amonia, n2_tese, "
                                         "n2_amonia_v45, n2_tese_v45, n2_amonia_a2, n2_tese_a2$"):
        cc.monta("n2_xyz", RAIZ / "casos")


def test_origem_ausente_dá_mensagem_clara(tmp_path):
    with pytest.raises(FileNotFoundError, match="casos/README.md"):
        cc.monta("n2_amonia", tmp_path)


@precisa_arquivo(*FONTES)
def test_linha_de_comando_escreve_no_destino(tmp_path):
    destino = tmp_path / "casos"
    for nome in ("xu2025", "ar_n2_o2", "n2"):
        (destino / nome).mkdir(parents=True)
        for f in ("kinet.inp", "bolsigdb.dat"):
            (destino / nome / f).write_text((RAIZ / "casos" / nome / f).read_text())
    r = subprocess.run([sys.executable, str(RAIZ / "ferramentas" / "casos_coluna.py"), "--casos", str(destino)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    for nome in cc.RECEITAS:
        assert (destino / nome / "kinet.inp").read_text() == (RAIZ / "casos" / nome / "kinet.inp").read_text()


# ---------------------------------------------------------------- N2(a'') separated from N2(C)

KINET_C = """ELEMENTS
N E
END
SPECIES
N2  N2(C3)  N2(B3)  E  N2^+
END
BOLSIG
N2
END
REACTIONS
e + N2 => e + N2(C3)                               !   BOLSIG N2 -> @R
    @R = N2(C3)  N2(E3)  N2(a''1)
N2(C3) => N2(B3)                                   !   2.7d7
END
"""


def test_a2_deixa_de_ir_para_o_n2_c_e_vira_espécie():
    novo = cc.separa_a2(KINET_C)
    linhas = novo.splitlines()
    i = linhas.index("    @R = N2(C3)  N2(E3)")
    assert linhas[i + 1].split("!")[0].split() == "e + N2 => e + N2(a``1)".split()
    assert linhas[i + 1].split("!")[1].strip() == "BOLSIG N2 -> N2(a''1)"
    # the species is N2(a``1): the preprocessor of ZDPlasKin refuses ''
    assert "N2(a``1)" in novo.split("SPECIES")[1].split("END")[0].split()
    # quenching by N2, Gordiets et al. 1995, table II, R29
    assert "N2(a``1) + N2 => N2 + N2 ! 1.0d-14" in novo
    assert novo.rstrip().endswith("END") and novo.count("\nEND") == 4


def test_a2_maiúscula_do_elétron_segue_a_da_reação():
    novo = cc.separa_a2(KINET_C.replace("e + N2 => e + N2(C3)", "E + N2 => E + N2(C3)"))
    assert "E + N2 => E + N2(a``1)" in novo


def test_a2_sem_a_lista_é_recusado():
    with pytest.raises(ValueError, match="^no @R list with N2\\(a''1\\) after an excitation to N2\\(C3\\)$"):
        cc.separa_a2(cc.separa_a2(KINET_C))


@precisa_arquivo(*FONTES)
@pytest.mark.parametrize("nome, origem", [("n2_amonia_a2", "n2_amonia"), ("n2_tese_a2", "n2_tese")])
def test_receita_a2_só_muda_o_a2(nome, origem):
    kinet, banco = cc.monta(nome, RAIZ / "casos")
    base, banco_base = cc.monta(origem, RAIZ / "casos")
    assert banco == banco_base
    novas = set(kinet.splitlines()) - set(base.splitlines())
    assert all("a''1" in l or "a``1" in l or l.startswith("#") or l.strip().startswith("@R") for l in novas), novas
    assert "N2(a``1)" in kinet.split("SPECIES")[1].split("END")[0].split()


@precisa_terceiros
def test_caso_a2_passa_no_pré_processador(tmp_path):
    import caso
    kinet, _ = cc.monta("n2_amonia_a2", RAIZ / "casos")
    (tmp_path / "kinet.inp").write_text(kinet)
    caso.gera_zdplaskin(tmp_path)
    assert "bolsig:N2->N2(A''1)" in (tmp_path / "zdplaskin_m.F90").read_text()
