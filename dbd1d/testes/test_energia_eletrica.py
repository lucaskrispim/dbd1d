# SPDX-License-Identifier: GPL-3.0-or-later
"""Electrical energy balance (discrete Poynting theorem): the power that the source gives
to the powered electrode, V dQ/dt, is the rate of change of the field energy in the gas
and in the barriers plus the work of the field on the charges, the sum over the faces of
J E d, the electrons and the ions. With the charges moved by the fluxes of the model and
the field of its Poisson equation the balance is exact up to rounding: the power computed
from the circuit (scripts/compara_stankov.py) is the work on the charges plus the change of
the field energy. It checks that the right-hand side moves the charges with the fluxes of
fluxos and the surface charges with what reaches the walls. The fluxes themselves are
tested in test_modelo.py, and the electron work that enters the energy equation
(ganho_de_energia) in test_energia.py; the Joule heating of the ions in the gas
temperature is not tested against this work."""
import numpy as np
import pytest

from conftest import RAIZ, quimica_do
from modelo import EPS0, Q, Modelo
from tabela import Tabela


@pytest.fixture(scope="module")
def pecas():
    return quimica_do(RAIZ), Tabela(RAIZ / "tabela_bolsig.dat")


def _energia_e_carga(md, t, y):
    """Field energy (J/m2) of gas and barriers and charge (C/m2) of the powered electrode."""
    n, s_esq, s_dir = md.separa(y)
    phi, campo = md.campo(t, n, s_esq, s_dir)
    v = md.tensao(t)
    c, c_dir = md.eps_d / md.espessura, md.eps_d_dir / md.espessura_dir
    u = 0.5 * EPS0 * (c * (v - phi[0]) ** 2 + md.eps_gas * np.sum(campo ** 2 * md.d) + c_dir * phi[-1] ** 2)
    return u, EPS0 * c * (v - phi[0])


@pytest.mark.parametrize("agrupamento, espessura_dir, semente", [(1.5, None, 3), (0.0, 1.0e-3, 5), (2.5, 2.0e-3, 7)])
def test_potencia_da_fonte_igual_a_energia_do_campo_mais_trabalho_nas_cargas(pecas, agrupamento, espessura_dir,
                                                                              semente):
    md = Modelo(*pecas, n_celulas=30, agrupamento=agrupamento, espessura=1.75e-3, espessura_dir=espessura_dir,
                v_pico=5.0e3, gama_sec=0.05)
    r = np.random.default_rng(semente)
    n = 10.0 ** r.uniform(8, 13, size=(md.n_esp, md.m))
    y = np.concatenate([n.ravel(), [1e-6, -2e-6]])
    t = 2.5e-8
    dy = md.rhs(t, y)
    # rate of change along the trajectory, by a central difference (the energy is quadratic in
    # the charges and in V, so the difference is exact but for the curvature of V(t))
    dt = 1e-14
    (u1, q1), (u0, q0) = _energia_e_carga(md, t + dt, y + dt * dy), _energia_e_carga(md, t - dt, y - dt * dy)
    du, dq = (u1 - u0) / (2 * dt), (q1 - q0) / (2 * dt)
    # work of the field on the charges, from the fluxes of the model at the faces
    n, s_esq, s_dir = md.separa(y)
    _, campo = md.campo(t, n, s_esq, s_dir)
    mob, dif, temp, _, _, _ = md.coeficientes(md.campo_nas_celulas(campo), n)
    G, _, _ = md.fluxos(n, campo, mob, dif, temp)
    corrente = Q * 1e6 * (md.carga @ G)                          # A/m2 at the faces
    trabalho = np.sum(corrente * campo * md.d)                     # W/m2
    fonte = md.tensao(t) * dq
    escala = max(abs(fonte), abs(trabalho), abs(du))
    assert du == pytest.approx(fonte - trabalho, abs=1e-9 * escala)
    # the electrons and the ions both do work here (a test that leaves one out fails)
    eletrons = np.sum(Q * 1e6 * md.carga[md.i_e] * G[md.i_e] * campo * md.d)
    assert abs(eletrons) > 1e-3 * escala and abs(trabalho - eletrons) > 1e-3 * escala
