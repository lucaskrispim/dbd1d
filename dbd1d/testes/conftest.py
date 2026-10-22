# SPDX-License-Identifier: GPL-3.0-or-later
import pytest
import pathlib
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


def precisa_arquivo(*caminhos):
    """Test that needs files of the research repository that the package does not
    distribute (ZDPlasKin code, LXCat cross sections, mechanisms of other authors,
    the old Fortran code and its references): skipped, before running, when one
    of them is missing."""
    faltam = [c for c in caminhos if not (RAIZ / c).exists()]
    return pytest.mark.skipif(bool(faltam), reason=f"needs {', '.join(faltam)}, which is not distributed "
                                                   "(casos/README.md)")


def precisa_caso(nome):
    """Test that runs the case casos/NAME: skipped when its chemistry or tables
    are not present (in the package, the third-party cases come without them)."""
    caso = RAIZ / "casos" / nome
    tem = ((caso / "zdplaskin_m.F90").exists() or (caso / "quimica.json").exists()) \
        and (caso / "tabela_energia_tg.dat").exists()
    return pytest.mark.skipif(not tem, reason=f"casos/{nome} without chemistry or tables (not distributed, casos/README.md)")
