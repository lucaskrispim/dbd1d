# SPDX-License-Identifier: GPL-3.0-or-later
"""Power per period (scripts/compara_stankov.py): the last periods one by one, for runs
that are not periodic (300 mbar), with the last one equal to that of ultimo_periodo."""
import json
import sys

import numpy as np
import pytest

from conftest import RAIZ

sys.path.insert(0, str(RAIZ / "scripts"))
import compara_stankov  # noqa: E402


def test_potencia_de_cada_um_dos_ultimos_periodos(tmp_path):
    # charge lagging the voltage by delta, amplitude growing period by period: the power of
    # period k is (1/T) * pi * V0 * Qk * sin(delta), with Q = eps0 eps_d (V - phi) A / Delta
    per, v0, delta = 4e-5, 1000.0, 0.3
    a = {"periodo": per, "eps_dieletrico": 4.2, "espessura": 1e-3}
    (tmp_path / "parametros.json").write_text(json.dumps({"argumentos": a}))
    t = np.linspace(0.0, 4 * per, 4 * 4000 + 1)
    k = np.minimum(np.floor(t / per + 1e-9), 3) + 1                      # period 1 to 4
    q0 = 1e-8 * k                                                        # C
    q = q0 * np.sin(2 * np.pi * t / per - delta)
    v = v0 * np.sin(2 * np.pi * t / per)
    fator = compara_stankov.EPS0 * a["eps_dieletrico"] / a["espessura"] * compara_stankov.AREA
    phi = v - q / fator
    np.savetxt(tmp_path / "superficie.dat", np.column_stack([t, v, phi, np.zeros_like(t), np.zeros_like(t)]),
               header="t V phi_esq phi_dir sigma")
    p = compara_stankov.potencias_por_periodo(tmp_path, 3)
    esperado = [np.pi * v0 * 1e-8 * n * np.sin(delta) / per for n in (2, 3, 4)]
    np.testing.assert_allclose(p, esperado, rtol=2e-3)
    assert p[-1] == pytest.approx(compara_stankov.ultimo_periodo(tmp_path)["P"], rel=1e-12)
