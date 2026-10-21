# SPDX-License-Identifier: GPL-3.0-or-later
import pytest
import pathlib
import re
import sys

PROTOTIPO = pathlib.Path(__file__).resolve().parent.parent
RAIZ = PROTOTIPO.parent
sys.path.insert(0, str(PROTOTIPO))


def pytest_configure(config):
    config.addinivalue_line("markers", "lento: gera tabelas com o BOLSIG+ (dezenas de segundos)")


def _tem_terceiros():
    """ZDPlasKin e BOLSIG+ disponíveis (ZDPLASKIN_DIR ou as cópias do repositório de pesquisa)."""
    sys.path.insert(0, str(RAIZ / "ferramentas"))
    import caso
    try:
        caso.terceiros()
        return True
    except caso.ErroCaso:
        return False


# testes que geram o zdplaskin_m.F90 ou as tabelas: pulados sem o ZDPlasKin e o
# BOLSIG+, que não vêm no pacote (a licença do ZDPlasKin não permite redistribuir)
precisa_terceiros = pytest.mark.skipif(not _tem_terceiros(), reason="ZDPlasKin e BOLSIG+ ausentes (defina ZDPLASKIN_DIR)")


def quimica_do(pasta):
    """Química do caso como o simula.py a escolhe: o zdplaskin_m.F90 quando existe
    (repositório de pesquisa), senão o quimica.json (pacote distribuído)."""
    from quimica_zdp import Quimica
    pasta = pathlib.Path(pasta)
    if (pasta / "zdplaskin_m.F90").exists():
        return Quimica(pasta / "zdplaskin_m.F90")
    return Quimica.de_json(pasta / "quimica.json")


# No pacote distribuído faltam, de propósito, os arquivos que não podemos
# redistribuir (código do ZDPlasKin, seções de choque do LXCat, mecanismos de
# outros autores) e as referências do código Fortran antigo. Um teste que falha
# só por um deles vira pulado, com o arquivo no motivo. No repositório de
# pesquisa (com o zdplaskin_m.F90 na raiz) nada muda.
NAO_DISTRIBUIDOS = [RAIZ / p for p in ("zdplaskin_m.F90", "bolsigdb.dat", "casos/n2", "casos/amonia_sun2022",
                                        "casos/amonia_sun2022_hongW98", "casos/ar_n2_o2", "casos/ar_n2",
                                        "casos/xu2025", "casos/yang2022_N2", "casos/yang2022_1pcH2",
                                        "referencias", "Makefile", "src", "eedf")]
PACOTE = not (RAIZ / "zdplaskin_m.F90").exists()


def _arquivo_nao_distribuido(texto):
    """Um caminho da mensagem que não existe e fica dentro de um arquivo ou
    pasta não distribuídos (os casos de terceiros têm no pacote só os arquivos
    do grupo: faltam a química e as tabelas)."""
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
    # só erro de arquivo ausente: FileNotFoundError, ou a saída de um subprocesso
    # (simula.py, scripts) que recusou um caso ou arquivo inexistente
    texto = call.excinfo.exconly()
    ausente = call.excinfo.errisinstance(FileNotFoundError) or any(
        s in texto for s in ("No such file", "não tem ", "não existe", "não é uma pasta"))
    falta = _arquivo_nao_distribuido(texto) if ausente else None
    if falta:
        rel.outcome = "skipped"
        rel.longrepr = (str(item.path), item.location[1] or 0,
                        f"precisa de {falta.relative_to(RAIZ)}, que não é distribuído (casos/LEIAME.md)")


def precisa_caso(nome):
    """Teste que roda o caso casos/NOME: pulado quando a química ou as tabelas
    dele não estão presentes (no pacote, os casos de terceiros vêm sem elas)."""
    caso = RAIZ / "casos" / nome
    tem = ((caso / "zdplaskin_m.F90").exists() or (caso / "quimica.json").exists()) \
        and (caso / "tabela_energia_tg.dat").exists()
    return pytest.mark.skipif(not tem, reason=f"casos/{nome} sem química ou tabelas (não distribuídos, casos/LEIAME.md)")
