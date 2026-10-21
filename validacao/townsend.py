# SPDX-License-Identifier: GPL-3.0-or-later
"""Verificação pelo critério de ruptura de Townsend: eletrodos metálicos, tensão
contínua, campo local (sem --energia), regime linear.

Referência: o problema linear contínuo que o modelo resolve, com coeficientes
constantes no campo uniforme E = V/d. Com s a distância ao cátodo, os elétrons
derivam para +s com w = mu E e difundem com D, e a ionização dá nu = k N:
    d/ds (w n - D dn/ds) = nu n  ->  n = soma c_j exp(lambda_j s),
    D lambda^2 - w lambda + nu = 0.
Os íons só derivam para o cátodo, e o fluxo deles no cátodo é a integral de
nu n. Condições de parede do modelo (modelo.fluxos), com o fator de Boltzmann da
meia célula igual a 1, que é o limite da malha fina:
    cátodo (s = 0): Gamma_e = gama Gamma_i - a_e n v_th/2
    ânodo  (s = d): Gamma_e = a_e n (v_th/2 + w)
com a_e = (1 - r)/(1 + r). A ruptura é o zero do determinante 2x2 em V.
O critério clássico gama (exp(alfa d) - 1) = 1, alfa = nu/w, fica como
comparação física: a reabsorção dos secundários pelo fluxo térmico no cátodo e a
difusão o deslocam.

Simulação: para cada tensão, o inventário de íons positivos no gap cresce ou
decai como exp(s t) depois do transitório; a taxa s é ajustada na janela final e
o zero de s(V) dá a tensão de ruptura discreta.

Uso: python3 validacao/townsend.py [--malhas 100 200 400 800] [--pd 10 15 20] [--python PY]
"""
import argparse
import functools
import json
import pathlib
import subprocess
import sys

import numpy as np
from scipy.optimize import brentq

RAIZ = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "prototipo"))

ME = 9.1093837015e-31
T_TRANSITORIO = 1.0e-5   # s: alguns trânsitos dos íons no gap de 1 a 2 mm a 100 Torr
KB = 1.380649e-23
TORR = 133.322


def raizes(w, d_e, nu):
    """Expoentes espaciais lambda_1 < lambda_2 de D l^2 - w l + nu = 0 (1/m)."""
    disc = np.sqrt(w * w - 4.0 * d_e * nu)
    return (w - disc) / (2.0 * d_e), (w + disc) / (2.0 * d_e)


def determinante(w, d_e, nu, v_th, gama, gap, a_e=1.0):
    """Determinante das condições de parede do problema linear contínuo; zero na
    ruptura. Com d_e -> 0 e v_th -> 0, recai no critério clássico."""
    l1, l2 = raizes(w, d_e, nu)
    # modos exp(l1 s) e exp(l2 (s - d)): o segundo, de expoente grande quando a
    # difusão é pequena, fica medido a partir do ânodo e não estoura
    no_catodo = (1.0, np.exp(-l2 * gap))
    no_anodo = (np.exp(l1 * gap), 1.0)
    ions = (nu * np.expm1(l1 * gap) / l1, nu * -np.expm1(-l2 * gap) / l2)   # fluxo no cátodo por modo
    m = np.zeros((2, 2))
    for j, lam in enumerate((l1, l2)):
        fluxo = w - d_e * lam                       # Gamma_e / n de cada modo
        m[0, j] = (fluxo + a_e * 0.5 * v_th) * no_catodo[j] - gama * ions[j]
        m[1, j] = (fluxo - a_e * (0.5 * v_th + w)) * no_anodo[j]
    return np.linalg.det(m)


def classico(alfa, gap, gama):
    return gama * np.expm1(alfa * gap) - 1.0


@functools.lru_cache(maxsize=None)
def modelo(gap, pressao):
    """Modelo montado como na simulação (caso do argônio, campo local)."""
    import simula
    a = simula.cria_parser().parse_args(["/dev/null", "--gap", repr(float(gap)), "--espessura", "0", "--pressao",
                                         repr(float(pressao)), "--n-celulas", "10"])
    return simula.monta_modelo(a, RAIZ, None)


def coeficientes_do_modelo(gap, pressao, v):
    """mu, D, T_e e nu do modelo montado como na simulação, no campo uniforme."""
    md = modelo(float(gap), float(pressao))
    campo = np.full(md.m, v / gap)
    n = np.zeros((md.n_esp, md.m))
    mob, dif, temp, k_b, e_n, extra = md.coeficientes(campo, n)
    k = md.q.coeficientes(k_b, extra["tg"], temp[md.i_e], e_n)
    ion = md.q.reacoes.index("bolsig:AR->AR^+")
    nu = k[ion, 0] * float(np.ravel(extra["n"])[0])     # 1/s
    return mob[md.i_e, 0], dif[md.i_e, 0], temp[md.i_e, 0], nu, e_n[0]


def ruptura_referencia(gap, pressao, gama, r=0.0, sem_termico=False, sem_difusao=False, v_ini=150.0, v_fim=3000.0):
    """Tensão de ruptura do problema contínuo e a do critério clássico (V)."""
    a_e = (1.0 - r) / (1.0 + r)

    def f(v, cla=False):
        mu, d_e, te, nu, _ = coeficientes_do_modelo(gap, pressao, v)
        w = mu * v / gap
        if cla:
            return classico(nu / w, gap, gama)
        v_th = 0.0 if sem_termico else np.sqrt(8.0 * KB * te / (np.pi * ME))
        return determinante(w, 1e-12 * d_e if sem_difusao else d_e, nu, v_th, gama, gap, a_e)

    vs = np.geomspace(v_ini, v_fim, 60)
    out = []
    for cla in (False, True):
        g = [f(v, cla) for v in vs]
        # abaixo de certo campo a raiz é complexa (w^2 < 4 D nu) e o determinante é NaN
        i = next(i for i in range(len(vs) - 1)
                 if np.isfinite(g[i]) and np.isfinite(g[i + 1]) and np.sign(g[i]) != np.sign(g[i + 1]))
        out.append(brentq(lambda v: f(v, cla), vs[i], vs[i + 1], xtol=1e-3))
    return tuple(out)


def roda(destino, gap, pressao, gama, v, celulas, py, t_end, reflexao=0.0):
    gap, pressao, gama, v = (float(x) for x in (gap, pressao, gama, v))
    cmd = [py, str(RAIZ / "prototipo" / "simula.py"), str(destino), "--gap", repr(gap), "--espessura", "0",
           "--pressao", repr(pressao), "--gama", repr(gama), "--n-celulas", str(celulas), "--agrupamento", "3.0",
           "--forma", "degrau", "--subida", "1e-9", "--v-pico", repr(v), "--neutro", "--ne0", "1e2",
           "--atol", "1e-14", "--t-end", repr(t_end), "--dt-saida", repr(t_end / 100),
           "--reflexao-eletrons", repr(float(reflexao))]
    # acima da ruptura a corrente cresce até o regime não linear, que eletrodos
    # metálicos não limitam, e o integrador pode parar: vale o trecho gravado antes
    subprocess.run(cmd, capture_output=True, env={"OPENBLAS_NUM_THREADS": "1", "PATH": ""})


def taxa_de_crescimento(destino):
    """Taxa s (1/s) do inventário de íons positivos, ajustada no último terço."""
    sys.path.insert(0, str(RAIZ / "scripts"))
    from resumo import blocos
    ts, inv = [], []
    for tb, nomes, linhas in blocos(destino / "perfil.dat"):
        a = np.array(linhas[1:-1])
        faces = [0.0]
        for c in a[:, 0]:
            faces.append(2.0 * c - faces[-1])
        h = np.diff(faces)
        ions = sum(a[:, nomes.index(s)] for s in ("AR^+", "AR2^+"))
        ts.append(tb)
        inv.append(float((ions * h).sum()))
    ts, inv = np.array(ts), np.array(inv)
    # regime linear: depois do transitório (um terço da janela) e enquanto o
    # inventário não passa de 1e4 vezes o inicial (a carga espacial deforma o campo)
    j = (ts >= T_TRANSITORIO) & (inv < 1e4 * inv[0])
    if np.count_nonzero(j) < 5:
        raise RuntimeError(f"{destino}: menos de 5 registros no regime linear")
    return np.polyfit(ts[j], np.log(inv[j]), 1)[0]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--malhas", type=int, nargs="+", default=[100, 200, 400, 800])
    p.add_argument("--pd", type=float, nargs="+", default=[10.0, 15.0, 20.0], help="p d em Torr cm")
    p.add_argument("--pressao-torr", type=float, default=100.0)
    p.add_argument("--gama", type=float, default=0.05)
    p.add_argument("--reflexao", type=float, default=0.0, help="reflexão dos elétrons na parede")
    p.add_argument("--t-end", type=float, default=3e-5)
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--saida", type=pathlib.Path, default=RAIZ / "execucoes" / "townsend")
    p.add_argument("--so-referencia", action="store_true")
    a = p.parse_args()
    pressao = a.pressao_torr * TORR
    res = []
    for pd in a.pd:
        gap = pd / a.pressao_torr * 1e-2
        v_ref, v_cla = ruptura_referencia(gap, pressao, a.gama, r=a.reflexao)
        linha = {"pd": pd, "gap": gap, "gama": a.gama, "reflexao": a.reflexao, "V_referencia": v_ref,
                 "V_classico": v_cla, "malhas": {}}
        print(f"p d = {pd} Torr cm: referência {v_ref:.1f} V, clássico {v_cla:.1f} V", flush=True)
        if not a.so_referencia:
            for m in a.malhas:
                vs = v_ref * np.array([0.97, 0.99, 1.01, 1.03])
                taxas = []
                for v in vs:
                    d = a.saida / f"g{a.gama:g}_r{a.reflexao:g}_pd{pd:g}_n{m}_V{v:.1f}"
                    roda(d, gap, pressao, a.gama, v, m, a.python, a.t_end, a.reflexao)
                    taxas.append(taxa_de_crescimento(d))
                c = np.polyfit(vs, taxas, 1)
                v_b = -c[1] / c[0]
                linha["malhas"][m] = {"V": v_b, "tensoes": vs.tolist(), "taxas": taxas}
                print(f"  {m} células: {v_b:.1f} V ({100 * (v_b / v_ref - 1):+.2f} % da referência)", flush=True)
        res.append(linha)
    a.saida.mkdir(parents=True, exist_ok=True)
    (a.saida / f"townsend_g{a.gama:g}_r{a.reflexao:g}.json").write_text(json.dumps(res, indent=2) + "\n")


if __name__ == "__main__":
    main()
