# SPDX-License-Identifier: GPL-3.0-or-later
"""Assembly of the public package (empacotamento/monta_pacote.py, research repository
only): a listed file that is not in git is reported, not dropped without notice."""
import sys

import pytest

from conftest import RAIZ, precisa_arquivo

pytestmark = precisa_arquivo("empacotamento/monta_pacote.py")


def _montador():
    sys.path.insert(0, str(RAIZ / "empacotamento"))
    import monta_pacote
    return monta_pacote


def test_padrao_sem_arquivo_rastreado_e_apontado():
    m = _montador()
    no_git = m.rastreados()
    assert RAIZ / "dbd1d" / "modelo.py" in no_git
    # without the tracked file, its pattern is reported
    sem = no_git - {RAIZ / "dbd1d" / "versao.py"}
    antes = m.INCLUI
    try:
        m.INCLUI = ["dbd1d/versao.py", "dbd1d/modelo.py"]
        assert m.padroes_vazios(sem) == ["dbd1d/versao.py"]
        assert m.fora_do_git(sem) == [RAIZ / "dbd1d" / "versao.py"]
        assert m.padroes_vazios(no_git) == [] and m.fora_do_git(no_git) == []
    finally:
        m.INCLUI = antes


def test_nenhum_padrao_sem_arquivo_no_disco():
    # every pattern of the list matches some file on disk (a typo in the list fails here;
    # whether the files are tracked is checked when the package is built)
    m = _montador()
    assert [p for p in m.INCLUI if not any(f.is_file() for f in RAIZ.glob(p))] == []


def test_montagem_para_com_padrao_vazio_ou_arquivo_fora_do_git(tmp_path, monkeypatch):
    m = _montador()
    sem = m.rastreados() - {RAIZ / "dbd1d" / "versao.py"}
    monkeypatch.setattr(m, "rastreados", lambda: sem)
    monkeypatch.setattr(m, "INCLUI", ["dbd1d/versao.py"])
    monkeypatch.setattr(sys, "argv", ["monta_pacote.py", str(tmp_path / "p")])
    with pytest.raises(SystemExit, match="padrões sem nenhum arquivo rastreado"):
        m.main()
    # a pattern with tracked files and one untracked: stops, unless accepted
    monkeypatch.setattr(m, "INCLUI", ["dbd1d/versao.py", "dbd1d/modelo.py"])
    monkeypatch.setattr(m, "padroes_vazios", lambda no_git=None: [])
    with pytest.raises(SystemExit, match="fora do git"):
        m.main()
    assert not (tmp_path / "p").exists()
