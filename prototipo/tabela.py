# SPDX-License-Identifier: GPL-3.0-or-later
"""Tabela dos coeficientes dos elétrons em E/N (tabela_bolsig.dat), com a mesma
interpolação do módulo Fortran tabela: linear em (ln E/N, ln valor), ou em
(ln E/N, valor) quando um dos nós vale zero, e valores presos nos extremos.

Colunas: Te (K), mu*N (1/(cm V s)), D*N (1/(cm s)), nu/N (cm3/s), Pel/N
(eV cm3/s) e as taxas de impacto eletrônico (cm3/s), na ordem de
bolsig_pointer do ZDPlasKin.
"""
import pathlib

import numpy as np

TE, MUN, DN, NUN, PEL = range(5)
K0 = 5


class Tabela:
    def __init__(self, arquivo=None, dados=None):
        if dados is None:
            linhas = [l.split() for l in pathlib.Path(arquivo).read_text().splitlines()
                      if l.strip() and not l.startswith("#")]
            dados = np.array([[float(x) for x in l] for l in linhas])
        self.e_n = dados[:, 0]
        self.valores = dados[:, 1:]
        self.ln_e_n = np.log(self.e_n)
        positivo = self.valores > 0.0
        self.ln_valores = np.where(positivo, np.log(np.where(positivo, self.valores, 1.0)), 0.0)
        self.positivo = positivo
        # avaliações acima do maior E/N da tabela (usam o último nó)
        self.fora = 0

    def __call__(self, e_n, t_gas=None):
        """Valores (células, colunas) em e_n (Td), vetor; a temperatura do gás
        não entra (tabela a 300 K)."""
        e_n = np.atleast_1d(np.asarray(e_n, dtype=float))
        self.fora += int(np.count_nonzero(e_n > self.e_n[-1]))
        x = np.log(np.clip(e_n, self.e_n[0], self.e_n[-1]))
        a = np.clip(np.searchsorted(self.ln_e_n, x, side="right") - 1, 0, len(self.e_n) - 2)
        t = ((x - self.ln_e_n[a]) / (self.ln_e_n[a + 1] - self.ln_e_n[a]))[:, None]
        y0, y1 = self.valores[a], self.valores[a + 1]
        ambos = self.positivo[a] & self.positivo[a + 1]
        log_interp = np.exp((1.0 - t) * self.ln_valores[a] + t * self.ln_valores[a + 1])
        lin_interp = (1.0 - t) * y0 + t * y1
        return np.where(ambos, log_interp, lin_interp)


# colunas de tabela_energia.dat depois de E/N
# (E_POT é a potência que o campo entrega, que no modelo de crescimento
# temporal do BOLSIG+ inclui k_ion*eps; a perda por colisões é E_PEL + E_PINEL;
# E_NUN é a frequência de colisão por N)
EPS, E_MUN, E_DN, E_MUEN, E_DEN, E_POT, E_PEL, E_PINEL, E_NUN, E_K0 = range(10)


class TabelaEnergia:
    """Coeficientes dos elétrons em função da energia média (eV), de
    tabela_energia.dat, para a equação de energia dos elétrons. Interpolação
    linear em (ln eps, ln valor), ou em (ln eps, valor) quando um dos nós vale
    zero; fora da faixa, valores presos nos extremos."""

    def __init__(self, arquivo=None, dados=None):
        if dados is None:
            linhas = [l.split() for l in pathlib.Path(arquivo).read_text().splitlines()
                      if l.strip() and not l.startswith("#")]
            dados = np.array([[float(x) for x in l] for l in linhas])
        self.e_n = dados[:, 0]
        self.eps = dados[:, 1]
        if np.any(np.diff(self.eps) <= 0.0):
            raise ValueError("energia media nao monotona em tabela_energia.dat")
        self.valores = dados[:, 1:]
        self.ln_eps = np.log(self.eps)
        self.positivo = self.valores > 0.0
        self.ln_valores = np.where(self.positivo, np.log(np.where(self.positivo, self.valores, 1.0)), 0.0)
        # avaliações acima da maior energia da tabela (usam o último nó) e
        # abaixo da menor
        self.fora = 0
        self.fora_abaixo = 0

    def __call__(self, eps, t_gas=None):
        """Coeficientes (células, colunas) nas energias médias dadas; a
        temperatura do gás não entra (tabela a 300 K)."""
        eps = np.atleast_1d(np.asarray(eps, dtype=float))
        self.fora += int(np.count_nonzero(eps > self.eps[-1] * (1.0 + 1e-12)))
        x = np.log(np.clip(eps, self.eps[0], self.eps[-1]))
        a = np.clip(np.searchsorted(self.ln_eps, x, side="right") - 1, 0, len(self.eps) - 2)
        t = ((x - self.ln_eps[a]) / (self.ln_eps[a + 1] - self.ln_eps[a]))[:, None]
        ambos = self.positivo[a] & self.positivo[a + 1]
        log_interp = np.exp((1.0 - t) * self.ln_valores[a] + t * self.ln_valores[a + 1])
        lin_interp = (1.0 - t) * self.valores[a] + t * self.valores[a + 1]
        return np.where(ambos, log_interp, lin_interp)

    def energia_de_campo_local(self, e_n, t_gas=None):
        """Energia média de regime no E/N dado (aproximação de campo local)."""
        e_n = np.clip(np.atleast_1d(e_n), self.e_n[0], self.e_n[-1])
        return np.exp(np.interp(np.log(e_n), np.log(self.e_n), self.ln_eps))

    def limites(self, t_gas):
        """Menor e maior energia média da tabela em cada célula."""
        forma = np.shape(np.atleast_1d(t_gas))
        return np.full(forma, self.eps[0]), np.full(forma, self.eps[-1])


class TabelaEnergiaTg:
    """Coeficientes dos elétrons por energia média e temperatura do gás, de
    tabela_energia_tg.dat (blocos por T_g, gerados pelo BOLSIG+ com a T_g do
    bloco; lacuna G4). Em cada nó de T_g, a interpolação em energia média da
    TabelaEnergia; entre os nós, linear em T_g. A menor energia média de cada
    nó é a de 1e-4 Td, ~(3/2) k T_g, e o piso entre nós é interpolado em T_g.
    Para os pisos dos dois nós casarem com o piso local, cada nó k é avaliado
    em eps + (piso_k - piso(T)) piso(T)/eps: o deslocamento leva o piso local ao
    do nó e some em energia alta; sem ele, perto do piso a perda elástica
    interpolada passava de 10 vezes a do BOLSIG+ (revisão). Fora da faixa de
    T_g, os valores do nó extremo (contados em fora_t)."""

    def __init__(self, arquivo):
        linhas = [l.split() for l in pathlib.Path(arquivo).read_text().splitlines()
                  if l.strip() and not l.startswith("#")]
        dados = np.array([[float(x) for x in l] for l in linhas])
        self.t_gas = np.unique(dados[:, 0])
        self.blocos = [TabelaEnergia(dados=dados[dados[:, 0] == t, 1:]) for t in self.t_gas]
        # compatibilidade com a TabelaEnergia: energias do nó de 300 K
        self.e_n, self.eps = self.blocos[0].e_n, self.blocos[0].eps
        self.eps_min = np.array([b.eps[0] for b in self.blocos])
        self.eps_max = np.array([b.eps[-1] for b in self.blocos])
        self.fora = 0
        self.fora_abaixo = 0
        self.fora_t = 0

    def _pesos(self, t_gas, n):
        t = np.broadcast_to(np.asarray(t_gas, dtype=float), (n,))
        self.fora_t += int(np.count_nonzero((t < self.t_gas[0]) | (t > self.t_gas[-1])))
        t = np.clip(t, self.t_gas[0], self.t_gas[-1])
        a = np.clip(np.searchsorted(self.t_gas, t, side="right") - 1, 0, len(self.t_gas) - 2)
        w = (t - self.t_gas[a]) / (self.t_gas[a + 1] - self.t_gas[a])
        return a, w

    def _em_t(self, f, x, t_gas):
        """(1 - w) f(a, x) + w f(a+1, x) por célula, só nos nós usados; f
        recebe o índice do nó, os valores e a seleção de células."""
        x = np.atleast_1d(np.asarray(x, dtype=float))
        a, w = self._pesos(t_gas, x.size)
        saida = None
        for k in np.unique(a):
            sel = a == k
            v0, v1 = f(k, x[sel], sel), f(k + 1, x[sel], sel)
            wk = w[sel].reshape((-1,) + (1,) * (np.ndim(v0) - 1))
            v = (1.0 - wk) * v0 + wk * v1
            if saida is None:
                saida = np.empty((x.size,) + np.shape(v)[1:])
            saida[sel] = v
        return saida

    def __call__(self, eps, t_gas):
        eps = np.atleast_1d(np.asarray(eps, dtype=float))
        t = np.broadcast_to(np.asarray(t_gas, dtype=float), eps.shape)
        piso = self.limites(t)[0]

        def no(k, e, sel):
            return self.blocos[k](e + (self.eps_min[k] - piso[sel]) * piso[sel] / e)
        return self._em_t(no, eps, t)

    def energia_de_campo_local(self, e_n, t_gas=None):
        return self._em_t(lambda k, e, sel: self.blocos[k].energia_de_campo_local(e), e_n,
                          self.t_gas[0] if t_gas is None else t_gas)

    def limites(self, t_gas):
        t = np.clip(np.atleast_1d(np.asarray(t_gas, dtype=float)), self.t_gas[0], self.t_gas[-1])
        return np.interp(t, self.t_gas, self.eps_min), np.interp(t, self.t_gas, self.eps_max)


class TabelaEEDF:
    """EEDF do BOLSIG+ em nós da tabela de energia (tabela_eedf.dat), para
    diagnóstico: devolve a EEDF numa energia média dada. A interpolação entre os
    dois nós vizinhos é na variável reduzida x = u/eps, com g(x) = eps^(3/2)
    f(x eps), linear em (ln eps, ln g) e com ln g extrapolado linearmente além da
    grade de cada nó; a forma de uma maxwelliana sai exata. A EEDF interpolada é
    renormalizada (integral de sqrt(u) f du = 1). Fora da faixa, o nó extremo."""

    def __init__(self, arquivo):
        blocos, atual = [], []
        for linha in pathlib.Path(arquivo).read_text().splitlines():
            if linha.startswith("#"):
                continue
            if linha.strip():
                atual.append([float(x) for x in linha.split()])
            elif atual:
                blocos.append(atual)
                atual = []
        if atual:
            blocos.append(atual)
        self.e_n = np.array([b[0][0] for b in blocos])
        self.eps = np.array([b[0][1] for b in blocos])
        self.u = [np.array(b[1:])[:, 0] for b in blocos]
        self.f = [np.array(b[1:])[:, 1] for b in blocos]
        self.ln_eps = np.log(self.eps)

    def _ln_g(self, k, x):
        """ln g do nó k em x, com extrapolação linear nas duas pontas."""
        xk = self.u[k] / self.eps[k]
        lg = 1.5 * self.ln_eps[k] + np.log(self.f[k])
        r = np.interp(x, xk, lg)
        abaixo, acima = x < xk[0], x > xk[-1]
        r[abaixo] = lg[0] + (x[abaixo] - xk[0]) * (lg[1] - lg[0]) / (xk[1] - xk[0])
        r[acima] = lg[-1] + (x[acima] - xk[-1]) * (lg[-1] - lg[-2]) / (xk[-1] - xk[-2])
        return r

    def __call__(self, eps):
        eps = float(np.clip(eps, self.eps[0], self.eps[-1]))
        x_alvo = np.log(eps)
        a = int(np.clip(np.searchsorted(self.ln_eps, x_alvo, side="right") - 1, 0, len(self.eps) - 2))
        t = (x_alvo - self.ln_eps[a]) / (self.ln_eps[a + 1] - self.ln_eps[a])
        x = self.u[a + 1] / self.eps[a + 1]
        g = np.exp((1.0 - t) * self._ln_g(a, x) + t * self._ln_g(a + 1, x))
        u, f = x * eps, g / eps ** 1.5
        return u, f / momentos_da_eedf(u, f)[0]


def momentos_da_eedf(u, f, sub=20):
    """Integrais de sqrt(u) f du e de u^(3/2) f du (norma e energia média) com
    ln f linear entre os pontos da grade (subdivididos) e, de 0 ao primeiro ponto,
    ln f extrapolado linearmente (f exponencial, exata para a maxwelliana)."""
    trapezio = getattr(np, "trapezoid", None) or np.trapz
    ln_f = np.log(f)
    inclinacao = (ln_f[1] - ln_f[0]) / (u[1] - u[0])
    inicio = np.linspace(0.0, u[0], sub * 4 + 1)[:-1]
    fino = [inicio]
    for i in range(len(u) - 1):
        fino.append(np.linspace(u[i], u[i + 1], sub + 1)[:-1])
    fino.append(u[-1:])
    uf = np.concatenate(fino)
    lf = np.where(uf < u[0], ln_f[0] + (uf - u[0]) * inclinacao, np.interp(uf, u, ln_f))
    ff = np.exp(lf)
    return trapezio(np.sqrt(uf) * ff, uf), trapezio(uf ** 1.5 * ff, uf)


class TabelaTg:
    """Coeficientes de campo local (colunas da Tabela) por E/N e temperatura do
    gás, montados dos blocos de tabela_energia_tg.dat (lacuna G4 em campo
    local): em cada nó de T_g, a interpolação da Tabela em E/N; entre os nós,
    linear em T_g no mesmo E/N. Te vem da energia média, (2/3) eps/k; as outras
    colunas são as do BOLSIG+ (a 300 K, idênticas às da tabela_bolsig.dat).
    Fora da faixa de T_g, o nó extremo (contado em fora_t)."""

    KB_EV = 1.380649e-23 / 1.602176634e-19

    def __init__(self, arquivo):
        linhas = [l.split() for l in pathlib.Path(arquivo).read_text().splitlines()
                  if l.strip() and not l.startswith("#")]
        dados = np.array([[float(x) for x in l] for l in linhas])
        self.t_gas = np.unique(dados[:, 0])
        self.blocos = []
        for t in self.t_gas:
            d = dados[dados[:, 0] == t, 1:]
            # E/N, Te, mu*N, D*N, nu/N, Pel/N, taxas (ordem da Tabela)
            lfa = np.column_stack([d[:, 0], d[:, 1 + EPS] / (1.5 * self.KB_EV), d[:, 1 + E_MUN],
                                   d[:, 1 + E_DN], d[:, 1 + E_NUN], d[:, 1 + E_PEL], d[:, 1 + E_K0:]])
            self.blocos.append(Tabela(dados=lfa))
        self.e_n = self.blocos[0].e_n
        self.fora = 0
        self.fora_t = 0

    def __call__(self, e_n, t_gas):
        e_n = np.atleast_1d(np.asarray(e_n, dtype=float))
        self.fora += int(np.count_nonzero(e_n > self.e_n[-1]))
        t = np.broadcast_to(np.asarray(t_gas, dtype=float), e_n.shape)
        self.fora_t += int(np.count_nonzero((t < self.t_gas[0]) | (t > self.t_gas[-1])))
        t = np.clip(t, self.t_gas[0], self.t_gas[-1])
        a = np.clip(np.searchsorted(self.t_gas, t, side="right") - 1, 0, len(self.t_gas) - 2)
        w = (t - self.t_gas[a]) / (self.t_gas[a + 1] - self.t_gas[a])
        saida = np.empty((e_n.size, self.blocos[0].valores.shape[1]))
        for k in np.unique(a):
            sel = a == k
            v0, v1 = self.blocos[k](e_n[sel]), self.blocos[k + 1](e_n[sel])
            saida[sel] = (1.0 - w[sel, None]) * v0 + w[sel, None] * v1
        return saida

