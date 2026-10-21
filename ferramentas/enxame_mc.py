# SPDX-License-Identifier: GPL-3.0-or-later
"""Monte Carlo electron swarm in argon, to check the two-term approximation of
BOLSIG+ (and hence tabela_bolsig.dat) with the same cross sections (bolsigdb.dat,
which is not distributed: download the SIGLO argon set from LXCat).

Usage:
    python3 ferramentas/enxame_mc.py maxwell         # verification with an exact solution
    python3 ferramentas/enxame_mc.py compare         # MC against the table, 100 to 2000 Td
    python3 ferramentas/enxame_mc.py check 1000      # another seed, 3 times longer relaxation
(the Portuguese names comparar and conferir are also accepted)

Model: uniform field, cold gas at rest (the table is at 300 K; the effect appears
only below ~10 Td), isotropic scattering, exact free flights with null
collisions (no time step). The comparison goes from 100 to 2000 Td: below that
the energy relaxes only by elastic collisions (loss 2m/M ~ 3e-5 per collision)
and would need millions of steps. The elastic momentum transfer cross section is
the effective one minus the inelastic ones, as BOLSIG+ treats the EFFECTIVE
block. In ionization, the remaining energy is shared at random between the two
electrons. The population grows in time (temporal growth) and is resampled at
random to keep its size, which does not change the averages.
"""
import pathlib
import sys

import numpy as np

Q = 1.602176634e-19        # C
ME = 9.1093837015e-31      # kg
KB = 1.380649e-23          # J/K
N_GAS = 101325.0 / (KB * 300.0)   # m-3, the density of the table
RAIZ = pathlib.Path(__file__).resolve().parent.parent


def le_secoes(arquivo):
    """Blocks of the LXCat format: returns {(type, reaction): (parameter, energy eV, cross section m2)}."""
    linhas = arquivo.read_text().splitlines()
    secoes = {}
    i = 0
    while i < len(linhas):
        tipo = linhas[i].strip()
        if tipo in ("EFFECTIVE", "ELASTIC", "EXCITATION", "IONIZATION", "ATTACHMENT"):
            reacao = linhas[i + 1].strip()
            if tipo == "ATTACHMENT":
                # no parameter line in the LXCat format
                parametro, j = 0.0, i + 2
            else:
                # the number can be glued to the slash: "2.9e-1/threshold energy"
                try:
                    parametro = float(linhas[i + 2].split("/")[0].split()[0])
                except (ValueError, IndexError):
                    raise ValueError(f"{arquivo}: {tipo} {reacao} without the parameter line") from None
                j = i + 3
            while not linhas[j].startswith("-----"):
                j += 1
            j += 1
            pontos = []
            while not linhas[j].startswith("-----"):
                pontos.append([float(x) for x in linhas[j].split()[:2]])
                j += 1
            e, s = np.array(pontos).T
            secoes[(tipo, reacao)] = (parametro, e, s)
            i = j
        i += 1
    return secoes


class Gas:
    """Cross sections of ground state argon: elastic, lumped excitation and ionization."""

    def __init__(self, secoes):
        razao, e_ef, s_ef = secoes[("EFFECTIVE", "Ar")]
        self.razao_massa = razao                         # m/M
        self.limiar_exc, e_ex, s_ex = secoes[("EXCITATION", "Ar <-> Ar*")]
        self.limiar_ion, e_io, s_io = secoes[("IONIZATION", "Ar -> Ar^+")]
        self.grade = np.unique(np.concatenate([e_ef, e_ex, e_io, np.geomspace(1e-4, 1e4, 5000)]))
        g = self.grade
        self.exc = np.interp(g, e_ex, s_ex, left=0.0, right=s_ex[-1])
        self.ion = np.interp(g, e_io, s_io, left=0.0, right=s_io[-1])
        efetiva = np.interp(g, e_ef, s_ef, right=s_ef[-1])
        self.el = np.maximum(efetiva - self.exc - self.ion, 0.0)

    def secoes(self, energia_ev):
        g = self.grade
        return (np.interp(energia_ev, g, self.el), np.interp(energia_ev, g, self.exc),
                np.interp(energia_ev, g, self.ion))


class Maxwell:
    """Constant elastic collision frequency nu0: sigma = nu0/(N v)."""

    def __init__(self, nu0, razao_massa):
        self.nu0 = nu0
        self.razao_massa = razao_massa
        self.limiar_exc = self.limiar_ion = np.inf

    def secoes(self, energia_ev):
        v = np.sqrt(2.0 * np.maximum(energia_ev, 1e-12) * Q / ME)
        z = np.zeros_like(energia_ev)
        return self.nu0 / (N_GAS * v), z, z


def isotropico(n, gerador):
    cos_t = 2.0 * gerador.random(n) - 1.0
    fi = 2.0 * np.pi * gerador.random(n)
    sen_t = np.sqrt(1.0 - cos_t ** 2)
    return np.stack([sen_t * np.cos(fi), sen_t * np.sin(fi), cos_t], axis=1)


def enxame(gas, e_n_td, n_eletrons=20000, nu_max=None, voos_relax=2000, voos_media=4000, semente=1,
           energia_inicial=1.0):
    """Steady-state averages by exact free flights with null collisions: drift
    velocity (m/s), mean energy (eV) and rate coefficients (m3/s) of excitation
    and ionization. Each electron draws its flight time with the maximum
    frequency nu_max; the flight is integrated analytically and the collision is
    real with probability nu(eps)/nu_max. The averages are temporal, summed over
    the flights: <eps> = sum(int eps dt)/sum(t), drift = sum(dz)/sum(t) and
    k = (collisions)/(N sum(t))."""
    gerador = np.random.default_rng(semente)
    campo = e_n_td * 1e-21 * N_GAS                       # V/m
    acel = Q * campo / ME                                # electrons accelerated towards +z
    if nu_max is None:
        e_teste = np.geomspace(1e-3, 1.0e4, 6000)
        s = sum(gas.secoes(e_teste))
        nu_max = 1.05 * np.max(N_GAS * s * np.sqrt(2.0 * e_teste * Q / ME))
    v = isotropico(n_eletrons, gerador) * np.sqrt(2.0 * energia_inicial * Q / ME)
    soma = {"t": 0.0, "dz": 0.0, "int_eps": 0.0, "n_exc": 0, "n_ion": 0}
    for voo in range(voos_relax + voos_media):
        tau = -np.log(1.0 - gerador.random(len(v))) / nu_max
        vz0 = v[:, 2].copy()
        v2 = np.sum(v * v, axis=1)
        v[:, 2] += acel * tau
        velocidade = np.linalg.norm(v, axis=1)
        energia = 0.5 * ME * velocidade ** 2 / Q
        if energia.max() > 1.0e4:
            raise SystemExit("energy above the range of the cross sections; raise nu_max")
        s_el, s_ex, s_io = gas.secoes(energia)
        r = gerador.random(len(v)) * nu_max / (N_GAS * velocidade)
        el = r < s_el
        ex = (~el) & (r < s_el + s_ex)
        io = (~el) & (~ex) & (r < s_el + s_ex + s_io)
        if voo >= voos_relax:
            soma["t"] += tau.sum()
            soma["dz"] += np.sum(vz0 * tau + 0.5 * acel * tau ** 2)
            soma["int_eps"] += np.sum(0.5 * ME * (v2 * tau + vz0 * acel * tau ** 2
                                                  + acel ** 2 * tau ** 3 / 3.0)) / Q
            soma["n_exc"] += int(ex.sum())
            soma["n_ion"] += int(io.sum())
        nova_energia = energia.copy()
        # isotropic elastic collision: loses 2(m/M)(1 - cos chi) of the energy
        dirs = isotropico(len(v), gerador)
        cos_chi = np.sum(dirs * v, axis=1) / np.maximum(velocidade, 1e-30)
        nova_energia[el] = energia[el] * (1.0 - 2.0 * gas.razao_massa * (1.0 - cos_chi[el]))
        nova_energia[ex] = energia[ex] - gas.limiar_exc
        sobra = energia[io] - gas.limiar_ion
        fracao = gerador.random(io.sum())
        nova_energia[io] = sobra * fracao
        colidiu = el | ex | io
        modulo = np.sqrt(2.0 * np.maximum(nova_energia[colidiu], 0.0) * Q / ME)
        v[colidiu] = dirs[colidiu] * modulo[:, None]
        # the electron created in ionization replaces a randomly drawn electron
        # (population of fixed size, temporal growth)
        if io.any():
            v_novo = isotropico(io.sum(), gerador) * np.sqrt(2.0 * sobra * (1.0 - fracao) * Q / ME)[:, None]
            v[gerador.choice(len(v), io.sum(), replace=False)] = v_novo
    t = soma["t"]
    return {"vz": soma["dz"] / t, "energia": soma["int_eps"] / t,
            "k_exc": soma["n_exc"] / (N_GAS * t), "k_ion": soma["n_ion"] / (N_GAS * t)}


def maxwell():
    """Constant frequency: mu = e/(m nu0) and <eps> = M e^2 E^2/(2 m^2 nu0^2) (cold
    gas), for any mass ratio; m/M = 1e-3 to relax in a few steps."""
    nu0 = 1.0e12
    razao = 1.0e-3
    gas = Maxwell(nu0, razao)
    e_n = 4.37                                           # <eps> ~ 1 eV
    campo = e_n * 1e-21 * N_GAS
    vd = Q * campo / (ME * nu0)
    energia = (1.0 / razao) * (Q * campo) ** 2 / (2.0 * ME * nu0 ** 2) / Q
    r = enxame(gas, e_n, nu_max=nu0, voos_relax=20000, voos_media=20000, n_eletrons=5000,
               energia_inicial=energia)
    print(f"drift:   MC {r['vz']:.4e}  exact {vd:.4e}  deviation {r['vz'] / vd - 1:+.2%}")
    print(f"energy:  MC {r['energia']:.4e}  exact {energia:.4e}  deviation {r['energia'] / energia - 1:+.2%}")
    ok = abs(r["vz"] / vd - 1) < 0.02 and abs(r["energia"] / energia - 1) < 0.02
    print("verification:", "ok" if ok else "FAILED")
    return ok


def tabela():
    linhas = [l.split() for l in (RAIZ / "tabela_bolsig.dat").read_text().splitlines() if not l.startswith("#")]
    return np.array([[float(x) for x in l] for l in linhas])


def comparar(lista=(100.0, 200.0, 300.0, 600.0, 1000.0, 2000.0), fator_relax=1, semente=1):
    gas = Gas(le_secoes(RAIZ / "bolsigdb.dat"))
    t = tabela()
    print(f"{'E/N':>6} {'muN MC/tab':>11} {'Te MC/tab':>10} {'k_exc MC/tab':>13} {'k_ion MC/tab':>13} {'alfa/N MC/tab':>14}")
    for e_n in lista:
        te0 = np.interp(np.log(e_n), np.log(t[:, 0]), t[:, 1])
        r = enxame(gas, e_n, energia_inicial=1.5 * KB * te0 / Q, voos_relax=2000 * fator_relax, semente=semente)
        campo = e_n * 1e-21 * N_GAS
        lin = [np.interp(np.log(e_n), np.log(t[:, 0]), t[:, j]) for j in range(t.shape[1])]
        mun_tab = lin[2]                                        # 1/(cm V s)
        mun_mc = r["vz"] / campo * N_GAS * 1e-2                 # (m2/(V s))*(m-3) -> 1/(cm V s)
        te_tab = lin[1]
        te_mc = r["energia"] * 2.0 / 3.0 * Q / KB
        k_exc_tab, k_ion_tab = lin[7] * 1e-6, lin[6] * 1e-6     # cm3/s -> m3/s
        alfa_tab = k_ion_tab / (mun_tab * 1e2 / N_GAS * campo)
        alfa_mc = r["k_ion"] / r["vz"]
        def razao(a, b):
            return a / b if b > 0 else float("nan")
        print(f"{e_n:6.0f} {razao(mun_mc, mun_tab):11.3f} {razao(te_mc, te_tab):10.3f} "
              f"{razao(r['k_exc'], k_exc_tab):13.3f} {razao(r['k_ion'], k_ion_tab):13.3f} {razao(alfa_mc, alfa_tab):14.3f}")


if __name__ == "__main__":
    modo = sys.argv[1] if len(sys.argv) > 1 else "compare"
    if modo == "maxwell":
        sys.exit(0 if maxwell() else 1)
    if modo in ("check", "conferir"):
        # the same E/N with another seed and a 3 times longer relaxation
        comparar(lista=[float(x) for x in sys.argv[2:]], fator_relax=3, semente=7)
    else:
        comparar()
