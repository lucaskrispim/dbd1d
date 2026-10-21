# SPDX-License-Identifier: GPL-3.0-or-later
import pytest
import pathlib
import re
import sys

PROTOTIPO = pathlib.Path(__file__).resolve().parent.parent
RAIZ = PROTOTIPO.parent
sys.path.insert(0, str(PROTOTIPO))


def pytest_configure(config):
    config.addinivalue_line("markers", "lento: generates tables with BOLSIG+ (tens of seconds)")


def _tem_terceiros():
    """ZDPlasKin and BOLSIG+ available (ZDPLASKIN_DIR or the copies of the research repository)."""
    sys.path.insert(0, str(RAIZ / "ferramentas"))
    import caso
    try:
        caso.terceiros()
        return True
    except caso.ErroCaso:
        return False


# tests that generate zdplaskin_m.F90 or the tables: skipped without ZDPlasKin and
# BOLSIG+, which do not come with the package (the ZDPlasKin licence does not allow redistribution)
precisa_terceiros = pytest.mark.skipif(not _tem_terceiros(), reason="ZDPlasKin and BOLSIG+ not found (set ZDPLASKIN_DIR)")


def quimica_do(pasta):
    """Chemistry of the case as simula.py chooses it: zdplaskin_m.F90 when it exists
    (research repository), otherwise quimica.json (distributed package)."""
    from quimica_zdp import Quimica
    pasta = pathlib.Path(pasta)
    if (pasta / "zdplaskin_m.F90").exists():
        return Quimica(pasta / "zdplaskin_m.F90")
    return Quimica.de_json(pasta / "quimica.json")


# The distributed package lacks, on purpose, the files we cannot
# redistribute (ZDPlasKin code, LXCat cross sections, mechanisms of
# other authors) and the references of the old Fortran code. A test that fails
# only because of one of them becomes skipped, with the file in the reason. In the research
# repository (with zdplaskin_m.F90 at the root) nothing changes.
NAO_DISTRIBUIDOS = [RAIZ / p for p in ("zdplaskin_m.F90", "bolsigdb.dat", "casos/n2", "casos/amonia_sun2022",
                                        "casos/amonia_sun2022_hongW98", "casos/ar_n2_o2", "casos/ar_n2",
                                        "casos/xu2025", "casos/yang2022_N2", "casos/yang2022_1pcH2",
                                        "referencias", "Makefile", "src", "eedf")]
PACOTE = not (RAIZ / "zdplaskin_m.F90").exists()


def _arquivo_nao_distribuido(texto):
    """A path of the message that does not exist and lies inside a file or
    folder that is not distributed (the third-party cases have in the package only
    the files of the group: the chemistry and the tables are missing)."""
    for m in re.finditer(re.escape(str(RAIZ)) + r"/[^\s'\":,)]*", texto):
        caminho = pathlib.Path(m.group(0))
        if caminho.exists():
            continue
        for base in NAO_DISTRIBUIDOS:
            if caminho == base or base in caminho.parents:
                return caminho
    return None


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    resultado = yield
    rel = resultado.get_result()
    if not (PACOTE and rel.failed and call.excinfo is not None):
        return
    # only a missing-file error: FileNotFoundError, or the output of a subprocess
    # (simula.py, scripts) that refused a nonexistent case or file
    texto = call.excinfo.exconly()
    ausente = call.excinfo.errisinstance(FileNotFoundError) or any(
        s in texto for s in ("No such file", "não tem ", "não existe", "não é uma pasta"))
    falta = _arquivo_nao_distribuido(texto) if ausente else None
    if falta:
        rel.outcome = "skipped"
        rel.longrepr = (str(item.path), item.location[1] or 0,
                        f"needs {falta.relative_to(RAIZ)}, which is not distributed (casos/LEIAME.md)")


def precisa_caso(nome):
    """Test that runs the case casos/NAME: skipped when its chemistry or tables
    are not present (in the package, the third-party cases come without them)."""
    caso = RAIZ / "casos" / nome
    tem = ((caso / "zdplaskin_m.F90").exists() or (caso / "quimica.json").exists()) \
        and (caso / "tabela_energia_tg.dat").exists()
    return pytest.mark.skipif(not tem, reason=f"casos/{nome} without chemistry or tables (not distributed, casos/LEIAME.md)")
