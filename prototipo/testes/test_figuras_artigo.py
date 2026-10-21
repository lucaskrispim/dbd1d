# SPDX-License-Identifier: GPL-3.0-or-later
"""Labels of the swarm figure of the article (artigo/figuras.py): proper names
capitalised and the temperature when the series is not at 300 K."""
import sys

import pytest

from conftest import RAIZ

pytest.importorskip("matplotlib")
sys.path.insert(0, str(RAIZ / "artigo"))
import figuras  # noqa: E402


@pytest.mark.parametrize("nome, rotulo", [
    ("ellis_1976_medida", "Ellis 1976"),
    ("kucukarpaci_lucas_1981_medida", "Kucukarpaci and Lucas 1981"),
    ("townsend_bailey_1922_77K_medida", "Townsend and Bailey 1922, 77 K"),
    ("escada_2007_calculada", "Escada 2007 (calc.)"),
])
def test_rotulo_enxame(nome, rotulo):
    assert figuras.rotulo_enxame(nome) == rotulo


def test_legenda_fica_abaixo_dos_dados_e_do_rotulo(tmp_path):
    # the legend covers neither a curve nor the x-axis label: it lies entirely below
    # the box of the panel with the numbers and the label
    import matplotlib.pyplot as plt
    fig, e = plt.subplots(1, 2, figsize=(7.2, 3.5))
    for x in e:
        x.plot([0, 1, 2], [0, 1e15, 4e15], label="uma curva com nome comprido")
        x.plot([0, 1, 2], [1e15, 2e15, 3e15], "--", label="outra")
        x.set(xlabel="t (µs)", ylabel="N (cm$^{-3}$)")
        figuras.legenda(x)
    figuras.fecha(fig, tmp_path / "f.pdf")
    r = fig.canvas.get_renderer()
    for x in e:
        leg = x.get_legend().get_window_extent(r)
        x.get_legend().set_visible(False)
        assert leg.y1 <= x.get_tightbbox(r).y0 + 0.5
    plt.close(fig)
