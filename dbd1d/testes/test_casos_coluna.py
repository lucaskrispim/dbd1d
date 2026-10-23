# SPDX-License-Identifier: GPL-3.0-or-later
"""Rebuilding the N2 chemistries of the positive column (ferramentas/casos_coluna.py)
from the cases they come from: the kinet.inp and bolsigdb.dat written are those of
the research cases, byte for byte."""
import subprocess
import sys

import pytest

from conftest import RAIZ, precisa_arquivo

sys.path.insert(0, str(RAIZ / "ferramentas"))
import casos_coluna as cc  # noqa: E402

FONTES = ("casos/xu2025/kinet.inp", "casos/xu2025/bolsigdb.dat", "casos/ar_n2_o2/kinet.inp",
          "casos/ar_n2_o2/bolsigdb.dat", "casos/n2/kinet.inp", "casos/n2/bolsigdb.dat")


def test_receitas_em_ordem_de_dependência():
    nomes = list(cc.RECEITAS)
    assert nomes == ["n2_exemplo", "n2_amonia", "n2_tese", "n2_amonia_v45", "n2_tese_v45"]
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
                                         "n2_amonia_v45, n2_tese_v45$"):
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
