# SPDX-License-Identifier: GPL-3.0-or-later
"""Coupling of the electron kinetics to the vibrational distribution
(ferramentas/acoplamento_vibracional.py): each N2(v) gets the cross sections of
N2 with the thresholds lowered by E_v, the reactions that use them, and the
BOLSIG+ mixture takes the populations of the ladder."""
import re
import sys

import pytest

from conftest import RAIZ, precisa_terceiros

sys.path.insert(0, str(RAIZ / "ferramentas"))
import acoplamento_vibracional as av  # noqa: E402
import escada_vibracional as ev  # noqa: E402

EV_K = 11604.518

BANCO = """ELASTIC
N2
1.950000e-5/mass ratio
SPECIES: e / N2
-----------------------------
 0.000000e+0\t1.000000e-20
 1.000000e+2\t1.000000e-20
-----------------------------

EXCITATION
N2 -> N2(v1)
2.900000e-1/threshold energy
SPECIES: e / N2
-----------------------------
 2.900000e-1\t0.000000e+0
 1.000000e+1\t1.000000e-21
-----------------------------

EXCITATION
N2 -> N2(A3,v0-4)
6.170000e+0/threshold energy
SPECIES: e / N2
-----------------------------
 6.170000e+0\t0.000000e+0
 2.000000e+1\t2.000000e-21
-----------------------------

IONIZATION
N2 -> N2^+
1.560000e+1/threshold energy
SPECIES: e / N2
-----------------------------
 1.560000e+1\t0.000000e+0
 1.000000e+2\t2.000000e-20
-----------------------------
"""


def blocos(texto):
    return [b for k, b in av.blocos(texto) if k == "bloco"]


def test_limiar_deslocado_pela_energia_vibracional():
    ev10 = ev.energia(10) / EV_K
    novo = av.banco_acoplado(BANCO, 10)
    ion = next(b for b in blocos(novo) if b[1] == "N2(v10) -> N2^+")
    assert float(ion[2].split("/")[0]) == pytest.approx(15.6 - ev10, rel=1e-6)
    dados = [tuple(map(float, l.split())) for l in ion if l[:1] == " "]
    # the cross section at eps is that of N2(X) at eps + E_v
    assert dados[-1] == pytest.approx((100.0 - ev10, 2e-20), rel=1e-6)


def test_elástica_copiada_e_vibração_não_copiada():
    novo = av.banco_acoplado(BANCO, 3)
    alvos = [b[1] for b in blocos(novo)]
    assert alvos.count("N2(v3)") == 1                       # elastic of N2(v3)
    assert not any(a.startswith("N2(v3) -> N2(v") for a in alvos)
    assert "N2 <-> N2(v1)" in alvos                         # superelastic inverse of the ground-state excitation


def test_processo_com_limiar_abaixo_do_mínimo_sai():
    # E_v of v = 40 is above the 6.17 eV of N2(A): that excitation leaves, the ionization stays
    novo = av.banco_acoplado(BANCO, 40)
    alvos = [b[1] for b in blocos(novo)]
    assert "N2(v40) -> N2(A3,v0-4)" not in alvos and "N2(v40) -> N2^+" in alvos
    assert av.limiar_deslocado(6.17, 40) is None and av.limiar_deslocado(15.6, 40) > 0.3


KINET = """ELEMENTS
N  E
END
SPECIES
N2  N2(v1)  N2(v2)  N2(A3)  N2^+  E
END
BOLSIG
N2
END
REACTIONS
E + N2 => E + N2(v1)                       ! BOLSIG N2 -> N2(v1)
E + N2 => E + N2(A3)                       ! BOLSIG N2 -> @R
    @R = N2(A3,v0-4)
E + N2 => E + E + N2^+                     ! BOLSIG N2 -> N2^+
N2(A3) + N2 => N2 + N2                     ! 3.0d-16
END
"""


def test_reações_a_partir_de_cada_nível():
    novo = av.kinet_acoplado(KINET, 2)
    assert "E + N2(v2) => E + E + N2^+ ! BOLSIG N2(v2) -> N2^+" in novo
    assert "E + N2(v1) => E + N2(A3) ! BOLSIG N2(v1) -> @R" in novo
    assert "E + N2(v1) => E + N2(v1)" not in novo               # no vibrational excitation from v > 0
    assert "BOLSIG\nN2  N2(v1)  N2(v2)\nEND" in novo
    longo = av.kinet_acoplado(KINET, 45)
    bloco = longo.split("BOLSIG\n")[1].split("\nEND")[0].splitlines()
    assert max(len(l) for l in bloco) < 100 and "N2(v45)" in bloco[-1]


def test_composição_vem_das_populações():
    texto = av.composicao({0: 0.7, 1: 0.2, 2: 0.1})
    linhas = dict(l.split() for l in texto.splitlines() if l and "traco" not in l)
    assert float(linhas["N2"]) == pytest.approx(0.7) and float(linhas["N2(v2)"]) == pytest.approx(0.1)


@precisa_terceiros
def test_caso_acoplado_passa_no_pré_processador(tmp_path):
    import caso
    base = (RAIZ / "casos" / "n2_amonia_v45" / "kinet.inp").read_text()
    (tmp_path / "kinet.inp").write_text(av.kinet_acoplado(base, 45))
    caso.gera_zdplaskin(tmp_path)
    assert "bolsig:N2(V30)->N2^+" in (tmp_path / "zdplaskin_m.F90").read_text()


def test_sem_deslocamento_copia_a_seção_de_N2_intacta():
    # decomposition of the coupling: N2(v) as a target with the cross sections of N2(X) as they are
    novo = av.banco_acoplado(BANCO, 40, deslocar=False, superelastica=False)
    exc = next(b for b in blocos(novo) if b[1] == "N2(v40) -> N2(A3,v0-4)")
    assert float(exc[2].split("/")[0]) == 6.17
    dados = [tuple(map(float, l.split())) for l in exc if l[:1] == " "]
    assert dados == [(6.17, 0.0), (20.0, 2e-21)]


def test_sem_superelástica_a_excitação_vibracional_fica_de_mão_única():
    alvos = [b[1] for b in blocos(av.banco_acoplado(BANCO, 3, superelastica=False))]
    assert "N2 -> N2(v1)" in alvos and "N2 <-> N2(v1)" not in alvos
    assert "N2(v3) -> N2^+" in alvos


def test_reações_sem_deslocamento_mantêm_todos_os_processos():
    banco = BANCO
    novo = av.kinet_acoplado(KINET, 40, banco, deslocar=False)
    assert "E + N2(v40) => E + N2(A3) ! BOLSIG N2(v40) -> @R" in novo
    assert "E + N2(v40) => E + N2(A3)" not in av.kinet_acoplado(KINET, 40, banco)


def test_meio_deslocamento_e_limiar_mínimo_configurável():
    # sensitivity of the coupling: half the shift E_v/2, and a lower minimum threshold
    ev10 = ev.energia(10) / EV_K
    novo = av.banco_acoplado(BANCO, 10, deslocar=0.5)
    ion = next(b for b in blocos(novo) if b[1] == "N2(v10) -> N2^+")
    assert float(ion[2].split("/")[0]) == pytest.approx(15.6 - 0.5 * ev10, rel=1e-6)
    # E_25 is 6.16 eV: N2(v25) -> N2(A) has a shifted threshold of 0.01 eV, below 0.3 and above 0.005
    assert av.limiar_deslocado(6.17, 25) is None
    assert av.limiar_deslocado(6.17, 25, minimo=0.005) == pytest.approx(6.17 - ev.energia(25) / EV_K, abs=0)
    alvos = [b[1] for b in blocos(av.banco_acoplado(BANCO, 25, minimo=0.005))]
    assert "N2(v25) -> N2(A3,v0-4)" in alvos
    assert "E + N2(v25) => E + N2(A3)" in av.kinet_acoplado(KINET, 25, BANCO, minimo=0.005)


# ---------------------------------------------------------------- edges and limits

def test_limiar_igual_ao_mínimo_fica_e_logo_abaixo_sai():
    ev5 = ev.energia(5) / EV_K
    minimo = 6.17 - ev5
    assert av.limiar_deslocado(6.17, 5, minimo=minimo) == minimo
    assert av.limiar_deslocado(6.17, 5, minimo=minimo + 1e-12) is None


def test_deslocamento_zero_e_inteiro():
    assert av.limiar_deslocado(15.6, 30, deslocar=0.0) == 15.6
    assert av.limiar_deslocado(15.6, 30, deslocar=1.0) == av.limiar_deslocado(15.6, 30)
    assert av.limiar_deslocado(15.6, 30) == pytest.approx(15.6 - ev.energia(30) / EV_K, rel=1e-14, abs=0)


@pytest.mark.parametrize("kw", [dict(deslocar=-0.1), dict(deslocar=1.1), dict(minimo=-0.1)])
def test_parâmetros_fora_da_faixa_são_recusados(kw):
    mensagem = (f"^deslocar is the fraction of E_v in the shift, between 0 and 1, not {kw['deslocar']}$"
                if "deslocar" in kw else f"^minimo, the minimum shifted threshold, cannot be negative \\({kw['minimo']}\\)$")
    with pytest.raises(ValueError, match=mensagem):
        av.banco_acoplado(BANCO, 3, **kw)
    with pytest.raises(ValueError, match=mensagem):
        av.kinet_acoplado(KINET, 3, BANCO, **kw)


def test_população_nula_não_entra_na_composição():
    texto = av.composicao({0: 0.9, 1: 0.1, 2: 0.0})
    assert "N2(v2)" not in texto and "N2(v1) 0.1" in texto


def test_composição_sem_população_é_recusada():
    with pytest.raises(ValueError, match="^composicao: no level with a positive population$"):
        av.composicao({0: 0.0, 1: 0.0})


def test_bloco_sem_as_linhas_de_traços_é_recusado():
    quebrado = "EXCITATION\nN2 -> N2(A3)\n6.17/threshold energy\nSPECIES: e / N2\n-----\n 6.17 0.0\n"
    with pytest.raises(ValueError, match="^block EXCITATION N2 -> N2\\(A3\\) without the two dashed lines of its table$"):
        av.blocos(quebrado)


def test_ponto_em_zero_abaixo_do_limiar_é_mantido_na_cópia():
    # some LXCat sets start the table at 0 eV with zero cross section, below the threshold
    banco = BANCO.replace(" 6.170000e+0\t0.000000e+0\n 2.000000e+1", " 0.000000e+0\t0.000000e+0\n"
                          " 6.170000e+0\t0.000000e+0\n 2.000000e+1")
    novo = av.banco_acoplado(banco, 2)
    exc = next(b for b in blocos(novo) if b[1] == "N2(v2) -> N2(A3,v0-4)")
    dados = [tuple(map(float, l.split())) for l in exc if l[:1] == " "]
    assert dados[0] == (0.0, 0.0) and dados[1][0] == pytest.approx(6.17 - ev.energia(2) / EV_K, rel=1e-6)


def test_cada_nível_tem_a_elástica_e_os_processos_válidos():
    vmax = 40
    novo = av.banco_acoplado(BANCO, vmax)
    alvos = [b[1] for b in blocos(novo)]
    for v in range(1, vmax + 1):
        do_nivel = [a for a in alvos if a.split("->")[0].strip() == f"N2(v{v})"]
        validos = [lim for lim in (6.17, 15.6) if av.limiar_deslocado(lim, v) is not None]
        assert len(do_nivel) == 1 + len(validos), v


# ---------------------------------------------------------------- regression of the text

def test_retrato_do_banco_e_do_kinet_acoplados_até_v_2():
    dados = RAIZ / "dbd1d" / "testes" / "dados"
    assert av.banco_acoplado(BANCO, 2) == (dados / "acoplado_v2_secoes_sinteticas.dat").read_text()
    assert av.kinet_acoplado(KINET, 2, BANCO) == (dados / "acoplado_v2_kinet.inp").read_text()


@precisa_terceiros
def test_caso_sem_deslocamento_passa_no_pré_processador(tmp_path):
    # the first step of the decomposition: N2(v) as targets with the cross sections of N2 as they are
    import caso
    base = (RAIZ / "casos" / "n2_amonia_v45" / "kinet.inp").read_text()
    banco = (RAIZ / "casos" / "n2_amonia" / "bolsigdb.dat").read_text()
    (tmp_path / "kinet.inp").write_text(av.kinet_acoplado(base, 45, banco, deslocar=0.0))
    caso.gera_zdplaskin(tmp_path)
    assert "BOLSIG:N2(V45)->N2(A3,V0-4)" in (tmp_path / "zdplaskin_m.F90").read_text().upper()



# ---------------------------------------------------------------- found by mutation testing

def test_limiar_mínimo_padrão_é_0_3_ev():
    assert av.LIMIAR_MINIMO == 0.3
    ev10 = ev.energia(10) / EV_K
    assert av.limiar_deslocado(ev10 + 0.31, 10) == pytest.approx(0.31, rel=1e-9)
    assert av.limiar_deslocado(ev10 + 0.29, 10) is None


def test_limiar_mínimo_zero_é_aceito():
    av.banco_acoplado(BANCO, 3, minimo=0.0)


ROTACAO = """ROTATION
N2 -> N2(rot)
2.000000e-2/threshold energy
SPECIES: e / N2
-----------------------------
 2.000000e-2\t0.000000e+0
 1.000000e+1\t1.000000e-20
-----------------------------

EFFECTIVE
N2
1.950000e-5/mass ratio
SPECIES: e / N2
-----------------------------
 0.000000e+0\t2.000000e-20
 1.000000e+2\t2.000000e-20
-----------------------------

ATTACHMENT
O2 -> O2^-
SPECIES: e / O2
-----------------------------
 0.000000e+0\t1.000000e-22
 1.000000e+1\t1.000000e-22
-----------------------------
"""


def test_rotação_e_efetiva_copiadas_sem_deslocamento():
    tipos = [b[0].strip() for b in blocos(ROTACAO)]
    assert tipos == ["ROTATION", "EFFECTIVE", "ATTACHMENT"]
    novo = blocos(av.banco_acoplado(ROTACAO, 30))
    rot = next(b for b in novo if b[1] == "N2(v30) -> N2(rot)")
    assert rot[2] == "2.000000e-2/threshold energy" and " 2.000000e-2\t0.000000e+0" in rot
    assert any(b[0].strip() == "EFFECTIVE" and b[1] == "N2(v30)" for b in novo)
    assert sum(b[1].startswith("O2") for b in novo) == 1        # other targets are not copied


def test_base_já_com_superelástica_dá_o_mesmo():
    assert av.banco_acoplado(BANCO.replace("N2 -> N2(v1)", "N2 <-> N2(v1)"), 3) == av.banco_acoplado(BANCO, 3)


def test_níveis_já_no_bloco_bolsig_não_se_repetem_e_linhas_de_oito():
    kinet = KINET.replace("BOLSIG\nN2\nEND", "BOLSIG\nN2\nN2(v1)  N2(v2)\nEND")
    bloco = av.kinet_acoplado(kinet, 20).split("BOLSIG\n")[1].split("\nEND")[0].splitlines()
    nomes = " ".join(bloco).split()
    assert nomes == ["N2"] + [f"N2(v{k})" for k in range(1, 21)]
    assert [len(l.split()) for l in bloco] == [8, 8, 5]


def test_processo_que_não_muda_a_espécie_não_é_copiado_para_os_níveis():
    kinet = KINET.replace("E + N2 => E + N2(v1)", "E + N2 => E + N2                           ! BOLSIG N2 -> N2(rot)\n"
                                                    "E + N2 => E + N2(v1)")
    assert "E + N2(v1) => E + N2 !" not in av.kinet_acoplado(kinet, 2)


def test_sem_banco_reações_só_até_vmax():
    novo = av.kinet_acoplado(KINET, 2)
    assert "N2(v2) -> N2^+" in novo and "N2(v3)" not in novo


def test_lista_arroba_r_com_vários_processos():
    kinet = KINET.replace("    @R = N2(A3,v0-4)", "    @R = N2(A3,v0-4)  N2(A3,v5-9)")
    assert "    @R = N2(A3,v0-4)  N2(A3,v5-9)" in av.kinet_acoplado(kinet, 2).split("E + N2(v1) => E + N2(A3)")[1]


def test_composição_termina_com_o_íon_de_traço():
    assert av.composicao({0: 0.6, 3: 0.4}).splitlines() == ["N2 0.6", "N2(v3) 0.4", "N2^+ traco"]



@pytest.mark.parametrize("final, alvo", [("EXCITATION\n", ""), ("EXCITATION\nN2 -> N2(B3)\n", "N2 -> N2(B3)")])
def test_bloco_truncado_no_fim_do_arquivo(final, alvo):
    with pytest.raises(ValueError, match=f"^block EXCITATION {re.escape(alvo)} without the two dashed lines of its table$"):
        av.blocos(BANCO + "\n" + final)
