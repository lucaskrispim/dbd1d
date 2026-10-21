# SPDX-License-Identifier: GPL-3.0-or-later
"""Modelo fluido 1D acoplado: transporte de todas as espécies, Poisson, química e
cargas superficiais da DBD num único sistema dy/dt = f(t, y), para um
integrador implícito de passo adaptativo (BDF do scipy).

Estado y: densidades (cm-3) das espécies transportadas, célula a célula, e as
cargas superficiais esquerda e direita (C/m2). O gás de fundo (uma espécie ou
mistura, gas.json) fica preso, com densidade constante, na T_g dada.

Discretização (a mesma geometria e as mesmas condições de parede do código
Fortran, achados N8, N16 a N20):
- malha agrupada nas paredes por tangente hiperbólica (agrupamento = 0: uniforme);
- Poisson nas células e nos nós das paredes, com a barreira dielétrica e a
  carga superficial na lei de Gauss de cada interface;
- fluxos internos de Scharfetter-Gummel com mobilidade e difusão das células e
  média harmônica nas faces;
- paredes absorventes: Gamma = n*(f*v_term/2 + a*w_n), f o fator de Boltzmann da
  meia célula quando a deriva se afasta da parede (N19); emissão secundária
  gama*Gamma_ions;
- química: reações do zdplaskin_m.F90 com as taxas de impacto eletrônico e Te
  da tabela em E/N, no campo do centro da célula.

Com energia=True, a densidade de energia dos elétrons n_eps (eV cm-3) entra no
estado (aproximação da energia média local, Hagelaar e Pitchford 2005): os
coeficientes dos elétrons (mobilidade, difusão, Te, taxas) vêm da energia média
n_eps/n_e e da T_g (tabela_energia_tg.dat por TabelaEnergiaTg, ou
tabela_energia.dat a 300 K), e
  d(n_eps)/dt + d(Gamma_eps)/dx = -Gamma_e*E - n_e*N*[(P_el + P_inel)/N](eps),
com Gamma_eps de Scharfetter-Gummel (mu_eps, D_eps, carga -1), o ganho pelo
campo calculado com os mesmos fluxos de elétrons do transporte e a perda pelas
potências elástica e inelástica tabeladas. A potência total P/N do BOLSIG+
não serve de perda: no crescimento temporal do BOLSIG+ ela inclui k_ion*eps, a
energia dos elétrons criados, que no fluido já aparece como diluição de n_eps
pelo aumento de n_e (contá-la de novo baixava a energia média e a ionização).
Na parede, cada elétron que sai por agitação térmica leva
(5/2)kTe, como em Hagelaar et al. 2000, e a deriva leva mu_eps*E*n_eps; os
secundários entram com eps_sec.

Com aquecimento=True, a temperatura do gás T (K) entra no estado, a volume
constante, como no código Fortran (termico.f90, cinetica.f90 e o ZDPlasKin com
GAS_HEATING):
  c_v N k_B dT/dt = d/dx(kappa(T) dT/dx) + Q_el + Q_J,
com N as partículas pesadas da célula, c_v e kappa(T) do gas.json (no argônio,
c_v = 3/2 e o kappa do DF do Fortran), média harmônica nas faces,
parede a t_parede a meia célula, Q_el a perda elástica dos elétrons (P_el/N da
tabela) e Q_J o Joule dos íons, e n_i mu_i E^2 (achado F5). T e N locais entram
nas taxas, na difusão dos íons (Einstein), na difusão dos neutros, nas mobilidades
iônicas e nos fluxos térmicos de parede. Com a tabela de energia em T_g
(TabelaEnergiaTg, lacuna G4), a T_g local entra também nos coeficientes dos
elétrons, na perda elástica e no piso da energia média; com a tabela a 300 K,
eles ficam os de 300 K.

Com gas_solto=True (gas_fixo = .false. no Fortran), o gás de fundo entra no
estado: difunde com o seu coeficiente do transporte.json, participa da
química e recebe de volta pela parede, como átomos neutros, as partículas que
ela absorve, cada uma com os seus atomos_do_fundo (achado N15; Ar2+ conta dois). O N dos elétrons (E/N, coeficientes da
tabela, perdas) passa a ser o dos neutros da célula, como o all_neutral do
ZDPlasKin.
"""
import copy
import json
import pathlib

import numpy as np
from scipy.linalg import solve_banded

from quimica_zdp import ErroQuimica
from tabela import E_DEN, E_DN, E_K0, E_MUEN, E_MUN, E_NUN, E_PEL, E_PINEL, E_POT, NUN, PEL

Q = 1.602176634e-19
EPS0 = 8.8541878128e-12
KB = 1.380649e-23
ME = 9.1093837015e-31
U = 1.66053906660e-27
N_LOSCHMIDT = 2.6867811e19     # cm-3


def bernoulli(x):
    """B(x) = x/(e^x - 1), com B(0) = 1."""
    x = np.asarray(x, dtype=float)
    pequeno = np.abs(x) < 1e-8
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        b = np.where(pequeno, 1.0 - 0.5 * x, x / np.expm1(np.where(pequeno, 1.0, x)))
    return np.where(np.isnan(b), 0.0, b)


def faces_tanh(m, comprimento, agrupamento):
    """Faces de m células em [0, comprimento], agrupadas nas duas paredes por
    tangente hiperbólica de intensidade agrupamento (0: malha uniforme)."""
    j = np.arange(m + 1)
    if agrupamento <= 0.0:
        return comprimento * j / m
    f = 0.5 * comprimento * (1.0 + np.tanh(agrupamento * (2.0 * j / m - 1.0)) / np.tanh(agrupamento))
    f[0], f[-1] = 0.0, comprimento
    return f


def condutividade_termica(t):
    """Condutividade térmica do argônio (W/(m K)), como DF no Fortran: a
    referência que o gas.json do argônio reproduz (test_gas, test_verificacao)."""
    k = (0.0307313 * t + 10.006) * 1e-3
    return np.where(t > 3000.0, k * np.sqrt(np.maximum(t, 3000.0) / 3000.0), k)


def media_harmonica(a, b):
    """2ab/(a + b), zero onde a + b é nulo (coeficiente nas faces)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        h = 2.0 * a * b / (a + b)
    return np.where((a + b) > 0.0, h, 0.0)


RAIZ = pathlib.Path(__file__).resolve().parent.parent


def _le_json(valor, nome):
    """Cópia do dicionário dado, do arquivo dado ou, sem valor, do arquivo
    nome na raiz do repositório."""
    if valor is None:
        valor = RAIZ / nome
    if not isinstance(valor, dict):
        valor = json.loads(pathlib.Path(valor).read_text())
    return copy.deepcopy(valor)


class ErroGas(ValueError):
    """Propriedades do gás de fundo ausentes ou inválidas (gas.json)."""


class ErroTransporte(ValueError):
    """Dados de transporte ausentes ou inválidos (transporte.json)."""


class Modelo:
    def __init__(self, quimica, tabela, n_celulas=100, gap=8.0e-4, agrupamento=0.0,
                 v_pico=1.0e3, periodo=1.0e-7, espessura=0.0, eps_dieletrico=4.8,
                 gama_sec=0.05, polarizabilidade=1.6411, sigma_cx=1.0e-18,
                 fundo=None, t_gas=300.0, n_gas=None,
                 tabela_energia=None, eps_sec=5.0, faces=None, freio=True,
                 aquecimento=False, t_parede=300.0, gas_solto=False, bolsigdb=None,
                 k0_ar=None, en_aquecimento_ar=None, k0_ar2=None, ganho_tres_corpos=0.0, eps_gas=1.0,
                 transporte=None, gas=None, tabela_calor=None,
                 forma="seno", subida=0.0, largura=0.0, descida=0.0,
                 espessura_dir=None, eps_dieletrico_dir=None, tabela_tensao=None):
        self.q, self.tab = quimica, tabela
        if quimica.bolsig_tres_corpos:
            raise ErroQuimica("reações do BOLSIG+ com dois elétrons reagentes: "
                              + ", ".join(quimica.bolsig_tres_corpos) + ". A taxa do BOLSIG+ é de dois corpos "
                              "(cm3/s) e não serve de coeficiente de três corpos; corrija o kinet.inp")
        # propriedades do gás de fundo (gas.json na raiz, ou o dado)
        self.gas = _le_json(gas, "gas.json")
        # pasta dos arquivos do caso (o simula.py a troca com --caso)
        self.caso = RAIZ
        fundos = self._le_gas(quimica, fundo)
        self.tab_e = tabela_energia
        self.energia = tabela_energia is not None
        self.eps_sec = eps_sec
        # fator de Boltzmann da meia célula no fluxo térmico de parede (N19); é um
        # modelo de subgrade que tende a 1 com a malha (revisão C2)
        self.freio = freio
        self.m = n_celulas
        self.L = gap
        self.base = faces_tanh(n_celulas, gap, agrupamento)             # malha de referência
        self.troca_malha(self.base if faces is None else np.asarray(faces, dtype=float))
        self._define_tensao(v_pico, periodo, forma, subida, largura, descida, tabela_tensao)
        self._define_barreiras(espessura, eps_dieletrico, espessura_dir, eps_dieletrico_dir)
        # permissividade relativa do gás no Poisson (T4: o Fortran usa 1,000516,
        # valor a 0 °C; a 300 K seria ~1,00047; o padrão 1 é o vácuo)
        self.eps_gas = eps_gas
        self.gama = gama_sec
        self.alfa, self.sigma_cx = polarizabilidade, sigma_cx
        self.t_gas = t_gas
        # densidade do gás a 1 atm na temperatura dada, cm-3
        self.n_gas = 101325.0 / (KB * t_gas) * 1e-6 if n_gas is None else n_gas
        self._define_especies(quimica, fundos, gas_solto)
        # dados de transporte por espécie (transporte.json na raiz, ou o dado)
        self.transporte = _le_json(transporte, "transporte.json")
        self._aplica_opcoes_do_argonio(k0_ar, en_aquecimento_ar, k0_ar2)
        self._le_massas(fundos)
        self._le_mobilidades()
        self._define_grupos(gas_solto)
        self.n_esp = len(self.especies)
        self.n_var = self.n_esp + (1 if self.energia else 0)
        if self.energia:
            self._prepara_energia(quimica, fundos, bolsigdb, ganho_tres_corpos)
        self.calor_reacao = self._calor_das_reacoes(quimica, ganho_tres_corpos)
        # calor rápido dos elétrons (processos do BOLSIG+ listados no gas.json,
        # como a excitação rotacional do N2): tabela_calor_tg.dat do caso
        self.tab_calor = None
        if self.gas.get("calor", {}).get("processos_rapidos"):
            if tabela_calor is None:
                raise ErroGas("gas.json lista processos_rapidos: falta a tabela_calor_tg.dat (ferramentas/caso.py "
                              "calor e tabelas) ou o parâmetro tabela_calor")
            self.tab_calor = tabela_calor
        self.aquecimento = aquecimento
        self.t_parede = t_parede
        if aquecimento:
            self.i_t = self.n_var
            self.n_var += 1
        self._le_difusao_e_perda(fundos[0][0], gas_solto)

    # ------------------------------------------------- etapas do construtor
    def _define_tensao(self, v_pico, periodo, forma, subida, largura, descida, tabela=None):
        """Forma da tensão: seno (padrão); degrau, subida linear em subida s e
        constante depois (tensão contínua); pulso, trapézio de subida, platô
        largura e descida, repetido a cada periodo; tabela, pares (t, V) dentro
        de um período (um pulso medido), interpolados e repetidos a cada
        periodo, com V nulo fora da tabela."""
        if forma not in ("seno", "degrau", "pulso", "tabela"):
            raise ValueError(f"forma da tensão desconhecida: {forma}")
        if forma == "pulso" and subida + largura + descida >= periodo:
            raise ValueError("pulso: subida + largura + descida precisa ser menor que o periodo")
        if forma == "tabela":
            t_tab, v_tab = (np.asarray(x, dtype=float) for x in tabela)
            if len(t_tab) < 2 or np.any(np.diff(t_tab) <= 0.0) or t_tab[0] < 0.0:
                raise ValueError("tabela da tensão: os instantes precisam ser crescentes, a partir de t >= 0")
            if t_tab[-1] >= periodo:
                raise ValueError("tabela da tensão: o último instante precisa ser menor que o periodo")
            # V nulo nas pontas: fora da tabela a tensão é zero, e um salto
            # daria corrente de deslocamento infinita
            if v_tab[0] != 0.0 or v_tab[-1] != 0.0:
                raise ValueError("tabela da tensão: precisa começar e terminar em zero")
            self.t_tab, self.v_tab = t_tab, v_tab
        self.v_pico, self.omega = v_pico, 2.0 * np.pi / periodo
        self.forma, self.periodo = forma, periodo
        self.subida, self.largura, self.descida = subida, largura, descida

    def _define_barreiras(self, espessura, eps_dieletrico, espessura_dir, eps_dieletrico_dir):
        """Barreira dielétrica de cada eletrodo; a direita, sem valor, igual à
        esquerda. Espessura zero é eletrodo metálico, e só nos dois lados."""
        self.espessura, self.eps_d = espessura, eps_dieletrico
        self.espessura_dir = espessura if espessura_dir is None else espessura_dir
        self.eps_d_dir = eps_dieletrico if eps_dieletrico_dir is None else eps_dieletrico_dir
        if (self.espessura > 0.0) != (self.espessura_dir > 0.0):
            raise ValueError("espessura: barreira só de um lado não é suportada (as duas ou nenhuma)")

    def _define_especies(self, quimica, fundos, gas_solto):
        """Espécies transportadas: todas menos as do gás de fundo, que só entra
        solto; numa mistura, cada espécie do fundo com a sua fração de N."""
        self.fundos = [(quimica.indice[nome], fracao) for nome, fracao in fundos]
        self.gas_solto = gas_solto
        indices_fundo = {i for i, _ in self.fundos}
        self.especies = [k for k in range(len(quimica.especies)) if gas_solto or k not in indices_fundo]
        self.nomes = [quimica.especies[k].upper() for k in self.especies]
        self.carga = quimica.carga[self.especies]
        # a mobilidade do transporte.json é a do próprio íon (v = K E): a
        # deriva usa só o sinal da carga, que entra inteira no Poisson
        self.sinal = np.sign(self.carga)
        self.i_e = self.nomes.index("E")

    def _aplica_opcoes_do_argonio(self, k0_ar, en_aquecimento_ar, k0_ar2):
        """Opções de linha de comando do argônio, que sobrepõem as mobilidades
        de Ar+ e Ar2+ do transporte.json."""
        opcoes = (("AR^+", "K0", k0_ar), ("AR^+", "EN_aquecimento", en_aquecimento_ar), ("AR2^+", "K0", k0_ar2))
        for nome, chave, valor in opcoes:
            if valor is None:
                continue
            if nome not in self.nomes:
                raise ErroTransporte(f"transporte: opção do argônio para {nome}, que não está na química")
            mob = self._objeto(nome, "mobilidade")
            if chave == "EN_aquecimento" and mob.get("tipo") != "frost":
                raise ErroTransporte(f"transporte: campo de aquecimento dado para {nome}, cuja mobilidade "
                                     f"não é frost ({mob.get('tipo')})")
            mob[chave] = valor

    def _le_massas(self, fundos):
        """Massa de cada espécie transportada e a do gás de fundo, para a troca
        elástica dos elétrons com ele; numa mistura, 1/M = soma de x_i/M_i
        (aproximação: pesa pela fração, e não pela frequência de colisão)."""
        self.massa = np.array([ME if s == "E" else self._numero(s, "massa_u") * U for s in self.nomes])
        if len(fundos) == 1:
            self.m_fundo = self._numero(fundos[0][0], "massa_u") * U
        else:
            self.m_fundo = 1.0 / sum(fracao / (self._numero(nome, "massa_u") * U) for nome, fracao in fundos)

    def _le_mobilidades(self):
        """Mobilidade de cada íon: constante, Frost ou tabela em E/N."""
        self._mob = {}
        for k, s in enumerate(self.nomes):
            if s == "E" or self.carga[k] == 0:
                continue
            mob = self._objeto(s, "mobilidade")
            if mob.get("tipo") == "tabela":
                self._confere_tabela_de_mobilidade(s, mob)
            else:
                faltam = {"constante": ["K0"], "frost": ["K0", "EN_aquecimento"]}.get(mob.get("tipo"))
                if faltam is None:
                    raise ErroTransporte(f"transporte: tipo de mobilidade {mob.get('tipo')} desconhecido ({s})")
                for chave in faltam:
                    self._numero(s, "mobilidade", chave)
            self._mob[k] = mob

    @staticmethod
    def _confere_tabela_de_mobilidade(s, mob):
        e_n, k0 = mob.get("E_N"), mob.get("K0")
        listas = isinstance(e_n, list) and isinstance(k0, list) and len(e_n) == len(k0) and len(e_n) >= 2
        if not listas or np.any(np.diff(e_n) <= 0) or np.any(np.asarray(k0, dtype=float) <= 0):
            raise ErroTransporte(f"transporte: {s} mobilidade em tabela precisa de E_N crescente e K0 "
                                 "positivo, do mesmo tamanho")
        if mob.get("acima", "preso") not in ("preso", "raiz"):
            raise ErroTransporte(f"transporte: {s} mobilidade em tabela: acima = {mob['acima']!r}, "
                                 "preso ou raiz")

    def _define_grupos(self, gas_solto):
        """Íons positivos, íons negativos, neutros, o íon inicial e, com o gás
        solto, os átomos que a parede devolve ao gás por partícula absorvida.
        Íons negativos: transporte e Joule como os positivos, sem emissão
        secundária; o campo das paredes os prende no volume (fluxos)."""
        nomes, carga = self.nomes, self.carga
        self.ions = [k for k in range(len(nomes)) if carga[k] > 0]
        self.ions_negativos = [k for k, s in enumerate(nomes) if carga[k] < 0 and s != "E"]
        ion = str(self.gas["ion_inicial"]).upper()
        if ion not in nomes or carga[nomes.index(ion)] <= 0:
            raise ErroGas(f"gas.json: ion_inicial {self.gas['ion_inicial']} não é íon positivo da química")
        self.ion_inicial = nomes.index(ion)
        self.neutros = [k for k, s in enumerate(nomes) if carga[k] == 0 and s != "E"]
        if gas_solto:
            # cada espécie do fundo no estado; atomos[j, k]: partículas do fundo j
            # que a parede devolve por partícula k absorvida (numa mistura,
            # atomos_do_fundo é um objeto por espécie do fundo, como
            # {"N2": 0.5, "H2": 1.5} para o NH3)
            self.i_fundos = [nomes.index(self.q.especies[i].upper()) for i, _ in self.fundos]
            nomes_fundo = [nomes[i] for i in self.i_fundos]
            self.atomos = np.zeros((len(self.i_fundos), len(nomes)))
            for k, s in enumerate(nomes):
                if s == "E" or k in self.i_fundos:
                    continue
                if len(nomes_fundo) == 1:
                    self.atomos[0, k] = self._numero(s, "atomos_do_fundo", faixa="não negativo")
                    continue
                self._objeto(s, "atomos_do_fundo")
                for j, f in enumerate(nomes_fundo):
                    self.atomos[j, k] = self._numero(s, "atomos_do_fundo", f, faixa="não negativo")

    def _prepara_energia(self, quimica, fundos, bolsigdb, ganho_tres_corpos):
        """Energia trocada pelos elétrons em cada reação fora da tabela: as
        colisões com espécies que não são o gás de fundo (no argônio, as do Ar*),
        ausentes da potência tabelada para o gás de fundo, e o ganho do elétron
        que sobra na recombinação a três corpos (dois elétrons reagem, um
        sobra). Hipótese do ganho: 0 (padrão) até a energia de ionização do gás
        (15,76 eV no argônio); ver ACHADOS. Com o bloco calor do gas.json, o que
        o elétron não leva vira calor do gás (limite superior: a radiação não é
        descontada). As reações do BOLSIG+ ficam de fora: a energia delas vem
        do bolsigdb.dat."""
        # química do quimica.json: os limiares vêm dele, nunca do bolsigdb.dat da
        # raiz (que é do argônio); lida do zdplaskin_m.F90 sem bolsigdb dado, o da raiz
        if bolsigdb is None and not getattr(quimica, "do_json", False):
            bolsigdb = RAIZ / "bolsigdb.dat"
        energia = quimica.energia_eletronica(bolsigdb, [nome for nome, _ in fundos])
        ie = quimica.indice["E"]
        tres = ((quimica.ordem[ie] >= 2) & (quimica.estequiometria[ie] == -1)
                & np.array([c is None for c in quimica.coluna_bolsig]))
        self.energia_reacao = energia + np.where(tres, ganho_tres_corpos, 0.0)

    def _le_difusao_e_perda(self, fundo, gas_solto):
        """Difusão dos neutros (como DK no Fortran), m2/s, na T e no N do gás, e
        fator de perda na parede: Chantry, gama/(2 - gama) do fluxo térmico
        n v/2 (1 para íons, elétrons e o gás de fundo, que a parede devolve)."""
        self._dif = {}
        self.fator_parede = np.ones(len(self.nomes))
        fundo_solto = set(self.i_fundos) if gas_solto else set()
        for k, s in enumerate(self.nomes):
            if s == "E" or self.carga[k] != 0:
                continue
            dif = self._objeto(s, "difusao")
            self._numero(s, "difusao", "D0")
            self._numero(s, "difusao", "T0")
            self._numero(s, "difusao", "expoente", faixa="finito")
            self._dif[k] = dif
            if k not in fundo_solto:
                gama = self._numero(s, "perda_na_parede", faixa="entre 0 e 1")
                self.fator_parede[k] = gama / (2.0 - gama)
        p_atm = (self.n_gas * 1e6) * (1.38e-23 * self.t_gas) / 101325.0
        self.difusao = np.zeros(len(self.nomes))
        for k, dif in self._dif.items():
            self.difusao[k] = dif["D0"] / p_atm * (self.t_gas / dif["T0"]) ** dif["expoente"]

    def _le_gas(self, quimica, fundo):
        """Valida o gas.json e devolve a lista (NOME, fração) do gás de fundo."""
        g = self.gas
        for chave in ("fundo", "cv_por_kb", "condutividade", "ion_inicial"):
            if chave not in g:
                raise ErroGas(f"gas.json: falta {chave}")
        if not isinstance(g["fundo"], dict) or not g["fundo"]:
            raise ErroGas(f"gas.json: fundo {g['fundo']!r}: dicionário de espécie e fração")
        fundos = []
        for nome, fracao in g["fundo"].items():
            if nome.upper() not in quimica.indice:
                raise ErroGas(f"gas.json: o gás de fundo {nome} não está na química")
            if not isinstance(fracao, (int, float)) or isinstance(fracao, bool) or not fracao > 0.0:
                raise ErroGas(f"gas.json: fração do gás de fundo {nome} = {fracao!r}: precisa ser positiva")
            if nome.upper() in [n for n, _ in fundos]:
                raise ErroGas(f"gas.json: o gás de fundo {nome.upper()} aparece duas vezes")
            fundos.append((nome.upper(), float(fracao)))
        if abs(sum(f for _, f in fundos) - 1.0) > 1e-9:
            raise ErroGas(f"gas.json: as frações do gás de fundo somam {sum(f for _, f in fundos):g}, e não 1")
        if fundo is not None and [fundo.upper()] != [n for n, _ in fundos]:
            raise ErroGas(f"gás de fundo {fundo} diferente do gas.json ({', '.join(n for n, _ in fundos)})")
        cv = g["cv_por_kb"]
        if not isinstance(cv, (int, float)) or isinstance(cv, bool) or not np.isfinite(cv) or cv <= 0.0:
            raise ErroGas(f"gas.json: cv_por_kb = {cv!r}: precisa ser número positivo")
        self.cv_por_kb = float(cv)
        c = g["condutividade"]
        if not isinstance(c, dict) or c.get("tipo") not in ("linear_com_raiz", "tabela"):
            raise ErroGas(f"gas.json: condutividade {c!r}: tipo linear_com_raiz ou tabela")
        faltam = [k for k in {"linear_com_raiz": ("a", "b", "escala", "t_raiz"), "tabela": ("T", "kappa")}[c["tipo"]]
                  if k not in c]
        if faltam:
            raise ErroGas(f"gas.json: condutividade sem {', '.join(faltam)}")
        if c["tipo"] == "tabela" and (len(c["T"]) != len(c["kappa"]) or len(c["T"]) < 2
                                      or np.any(np.diff(c["T"]) <= 0) or np.any(np.asarray(c["kappa"]) <= 0)):
            raise ErroGas("gas.json: condutividade em tabela precisa de T crescente e kappa positivo, do mesmo tamanho")
        return fundos

    def _calor_das_reacoes(self, quimica, ganho_tres_corpos):
        """Calor (eV por evento) de cada reação, -soma(nu h), com as entalpias do
        bloco calor do gas.json; None sem o bloco. Ficam de fora as reações do
        BOLSIG+ (a energia sai dos elétrons, em P_inel) e as de um só reagente
        (radiativas: o fóton sai). Na recombinação a três corpos, o ganho do
        elétron que sobra é descontado."""
        calor = self.gas.get("calor")
        if calor is None:
            return None
        ent = {k.upper(): v for k, v in calor.get("entalpias", {}).items()}
        ent["E"] = 0.0
        ie = quimica.indice["E"]
        nu, ordem = quimica.estequiometria, quimica.ordem
        q = np.zeros(len(quimica.reacoes))
        faltam = set()
        for r in range(len(quimica.reacoes)):
            if quimica.coluna_bolsig[r] is not None or ordem[:, r].sum() <= 1:
                continue
            for s in np.nonzero(nu[:, r])[0]:
                nome = quimica.especies[s].upper()
                if nome not in ent:
                    faltam.add(nome)
                else:
                    q[r] -= nu[s, r] * ent[nome]
            if ordem[ie, r] >= 2 and nu[ie, r] == -1:
                q[r] -= ganho_tres_corpos
        if faltam:
            raise ErroGas(f"gas.json: falta a entalpia de {', '.join(sorted(faltam))} no bloco calor")
        self._eletrons_perdidos = np.where(np.array([c is None for c in quimica.coluna_bolsig]),
                                           np.minimum(nu[ie], 0.0), 0.0)
        return q

    def condutividade(self, t):
        """Condutividade térmica do gás (W/(m K)) na temperatura t (K)."""
        c = self.gas["condutividade"]
        t = np.asarray(t, dtype=float)
        if c["tipo"] == "tabela":
            return np.interp(t, c["T"], c["kappa"])
        k = (c["a"] * t + c["b"]) * c["escala"]
        return np.where(t > c["t_raiz"], k * np.sqrt(np.maximum(t, c["t_raiz"]) / c["t_raiz"]), k)

    def _dado(self, especie, campo):
        """Campo dos dados de transporte da espécie; erro explícito se falta."""
        if especie not in self.transporte:
            raise ErroTransporte(f"transporte: sem dados da espécie {especie}")
        if campo not in self.transporte[especie]:
            raise ErroTransporte(f"transporte: a espécie {especie} não tem {campo}")
        return self.transporte[especie][campo]

    def _objeto(self, especie, campo):
        valor = self._dado(especie, campo)
        if not isinstance(valor, dict):
            raise ErroTransporte(f"transporte: {especie} {campo} = {valor!r}: precisa ser um objeto")
        return valor

    FAIXAS = {"positivo": lambda v: v > 0.0, "não negativo": lambda v: v >= 0.0,
              "entre 0 e 1": lambda v: 0.0 <= v <= 1.0, "finito": lambda v: True}

    def _numero(self, especie, campo, chave=None, faixa="positivo"):
        """Número finito do campo (ou campo.chave) da espécie, na faixa dada."""
        if chave:
            alvo = self._objeto(especie, campo)
            if chave not in alvo:
                raise ErroTransporte(f"transporte: a espécie {especie} não tem {campo}.{chave}")
            valor = alvo[chave]
        else:
            valor = self._dado(especie, campo)
        numero = isinstance(valor, (int, float)) and not isinstance(valor, bool) and np.isfinite(valor)
        if not (numero and self.FAIXAS[faixa](valor)):
            nome = f"{campo}.{chave}" if chave else campo
            raise ErroTransporte(f"transporte: {especie} {nome} = {valor!r}: precisa ser número {faixa}")
        return float(valor)

    def troca_malha(self, faces):
        """Usa a malha de faces dadas (mesmo número de células, extremos 0 e L)."""
        if len(faces) != self.m + 1 or faces[0] != 0.0 or abs(faces[-1] - self.L) > 1e-15 * self.L:
            raise ValueError("malha incompatível com o modelo")
        if np.any(np.diff(faces) <= 0.0):
            raise ValueError("faces da malha fora de ordem")
        self.faces = np.array(faces, dtype=float)
        self.faces[-1] = self.L
        self.h = np.diff(self.faces)                                     # larguras
        self.xc = 0.5 * (self.faces[1:] + self.faces[:-1])               # centros
        nos = np.concatenate([[0.0], self.xc, [self.L]])
        self.d = np.diff(nos)                                            # m+1 distâncias entre nós

    # ------------------------------------------------------------------ campo
    def tensao(self, t):
        """Tensão aplicada no eletrodo esquerdo (V) no instante t (s)."""
        if self.forma == "seno":
            return self.v_pico * np.sin(self.omega * t)
        if self.forma == "degrau":
            return self.v_pico * min(t / self.subida, 1.0)
        s = t % self.periodo
        if self.forma == "tabela":
            return float(np.interp(s, self.t_tab, self.v_tab, left=0.0, right=0.0))
        if s < self.subida:
            return self.v_pico * s / self.subida
        if s < self.subida + self.largura:
            return self.v_pico
        if s < self.subida + self.largura + self.descida:
            return self.v_pico * (self.subida + self.largura + self.descida - s) / self.descida
        return 0.0

    def poisson(self, rho, sigma_esq, sigma_dir, v):
        """phi nos nós (paredes e centros) e campo nas m+1 faces. rho em C/m3."""
        m, d = self.m, self.d
        n = m + 2
        ab = np.zeros((3, n))       # forma em banda do solve_banded: superior, diagonal, inferior
        rhs = np.zeros(n)
        # células: (phi_{i+1}-phi_i)/d_i - (phi_i-phi_{i-1})/d_{i-1} = -rho h/eps0
        for_cel = np.arange(1, m + 1)
        g_w = self.eps_gas / d[for_cel - 1]
        g_e = self.eps_gas / d[for_cel]
        ab[1, for_cel] = -(g_w + g_e)
        ab[0, for_cel + 1] = g_e
        ab[2, for_cel - 1] = g_w
        rhs[for_cel] = -rho * self.h / EPS0
        if self.espessura > 0.0:
            c = self.eps_d / self.espessura
            c_dir = self.eps_d_dir / self.espessura_dir
            # c*(V - phi_0) + (phi_1 - phi_0)/d_0 + sigma/eps0 = 0
            ab[1, 0] = -(c + self.eps_gas / d[0])
            ab[0, 1] = self.eps_gas / d[0]
            rhs[0] = -c * v - sigma_esq / EPS0
            # -c_dir*phi_n + (phi_{n-1} - phi_n)/d_m + sigma/eps0 = 0
            ab[1, n - 1] = -(c_dir + self.eps_gas / d[-1])
            ab[2, n - 2] = self.eps_gas / d[-1]
            rhs[n - 1] = -sigma_dir / EPS0
        else:
            ab[1, 0], rhs[0] = 1.0, v
            ab[1, n - 1], rhs[n - 1] = 1.0, 0.0
        phi = solve_banded((1, 1), ab, rhs)
        campo = -np.diff(phi) / d
        return phi, campo

    def campo_nas_celulas(self, campo):
        """Campo nos centros das células, interpolado do campo nas faces."""
        nos = np.concatenate([[0.0], self.xc, [self.L]])
        x_f = 0.5 * (nos[1:] + nos[:-1])                 # ponto médio de cada par de nós
        w = (self.xc - x_f[:-1]) / (x_f[1:] - x_f[:-1])
        return (1.0 - w) * campo[:-1] + w * campo[1:]

    # ------------------------------------------------------------ coeficientes
    # mobilidades reduzidas medidas (cm2/(V s), na densidade de Loschmidt),
    # constantes ou pela fórmula de Frost, K = K0*(1 + (E/N)/(E/N)_aq)**(-1/2);
    # valores e fontes no transporte.json

    def mobilidade_ion(self, k, campo, n_pesadas=None):
        """Mobilidade (m2/(V s)) do íon k nas células, a partir das medidas, com
        mu = K*N0/N; N é o do gás ou, se dado, o das partículas pesadas."""
        mob = self._mob[k]
        escala = N_LOSCHMIDT / (self.n_gas if n_pesadas is None else n_pesadas) * 1e-4
        if mob["tipo"] == "constante":
            return np.full_like(np.asarray(campo, dtype=float), mob["K0"] * escala)
        e_n = np.abs(campo) / (self.n_gas if n_pesadas is None else n_pesadas) * 1e15
        if mob["tipo"] == "tabela":
            # K0 medido por E/N, linear; abaixo, preso; acima, preso ou com a cauda
            # de campo alto K ~ (E/N)^(-1/2) a partir do último ponto medido
            k0 = np.interp(e_n, mob["E_N"], mob["K0"])
            if mob.get("acima", "preso") == "raiz":
                ultimo = mob["E_N"][-1]
                k0 = np.where(e_n > ultimo, mob["K0"][-1] * np.sqrt(ultimo / np.maximum(e_n, ultimo)), k0)
            return k0 * escala
        return mob["K0"] * escala / np.sqrt(1.0 + e_n / mob["EN_aquecimento"])

    def n_pesadas(self, n):
        """Partículas pesadas por célula (cm-3): o gás de fundo e as espécies que
        não são elétrons, como MDT no Fortran."""
        pesadas = [k for k in range(self.n_esp) if k != self.i_e]
        soma = np.maximum(n[pesadas], 0.0).sum(axis=0)
        return soma if self.gas_solto else self.n_gas + soma

    def n_neutro(self, n):
        """N dos elétrons (cm-3): o do gás com o gás preso; com o gás solto, os
        neutros da célula (o gás de fundo e os neutros transportados), como o
        all_neutral do ZDPlasKin."""
        if not self.gas_solto:
            return self.n_gas
        return np.maximum(n[self.neutros], 0.0).sum(axis=0)

    def coeficientes(self, campo_cel, n=None, n_eps=None, t_gas=None):
        """Mobilidade (m2/(V s)), difusão (m2/s) e temperatura (K) de cada espécie
        nas células, taxas de impacto eletrônico (células, colunas), E/N (Td) e
        os coeficientes extras: P_el/N (eV cm3/s) e, com a equação de energia, os
        de energia dos elétrons. Com t_gas (K, por célula), a temperatura e as
        partículas pesadas locais entram nos coeficientes dos íons e dos neutros."""
        n_e_neutro = self.n_neutro(n)
        e_n = np.abs(campo_cel) / n_e_neutro * 1e15
        mob = np.zeros((self.n_esp, self.m))
        dif = np.zeros((self.n_esp, self.m))
        local = t_gas is not None or self.gas_solto
        tg = np.full(self.m, self.t_gas) if t_gas is None else np.asarray(t_gas, dtype=float)
        # a mobilidade iônica usa sempre o N das partículas pesadas da célula,
        # como ions.f90; a difusão do Ar*, só com aquecimento ou gás solto
        n_pes = self.n_pesadas(n) if n is not None else None
        temp = np.tile(tg, (self.n_esp, 1))
        extra = {}
        if self.energia:
            eps, eps_bruta = self.energia_media(n, n_eps, e_n, tg, com_bruta=True)
            v = self.tab_e(eps, tg)
            # elétrons abaixo do piso da tabela (mais frios que o gás): a tabela
            # não os tem (o BOLSIG+ dá o regime com campo), e o valor no piso
            # faria o gás seguir esquentando; a perda elástica ganha a relaxação
            # (2m/M)(nu/N)(eps - piso) e fica negativa, levando-os ao gás
            pel = v[:, E_PEL]
            abaixo = eps_bruta < eps
            if np.any(abaixo):
                pel = pel + np.where(abaixo, 2.0 * ME / self.m_fundo * v[:, E_NUN] * (eps_bruta - eps), 0.0)
            mob[self.i_e] = v[:, E_MUN] / n_e_neutro * 1e-4
            dif[self.i_e] = v[:, E_DN] / n_e_neutro * 1e-4
            temp[self.i_e] = (2.0 / 3.0) * eps * Q / KB
            k_bolsig = v[:, E_K0:]
            extra = {"eps": eps, "k_bolsig": k_bolsig, "mu_eps": v[:, E_MUEN] / n_e_neutro * 1e-4,
                     "d_eps": v[:, E_DEN] / n_e_neutro * 1e-4, "pot": v[:, E_POT],
                     "pel": pel, "perda": pel + v[:, E_PINEL],
                     "nu": v[:, E_NUN]}
        else:
            v = self.tab(e_n, tg)
            mob[self.i_e] = v[:, 1] / n_e_neutro * 1e-4
            dif[self.i_e] = v[:, 2] / n_e_neutro * 1e-4
            temp[self.i_e] = v[:, 0]
            k_bolsig = v[:, 5:]
            extra = {"pel": v[:, PEL], "nu": v[:, NUN]}
        extra["n"] = n_e_neutro
        extra["tg"] = tg
        extra["e_n"] = e_n                      # E/N local (Td), para as taxas que usam EN
        for k in range(self.n_esp):
            if k == self.i_e:
                continue
            if self.carga[k] != 0:
                mob[k] = self.mobilidade_ion(k, campo_cel, n_pes)
                dif[k] = mob[k] * KB * tg / (abs(self.carga[k]) * Q)       # Einstein, carga z
            elif local:
                p_atm = (n_pes * 1e6) * (1.38e-23 * tg) / 101325.0
                d = self._dif[k]
                dif[k] = d["D0"] / p_atm * (tg / d["T0"]) ** d["expoente"]
            else:
                dif[k] = self.difusao[k]
        return mob, dif, temp, k_bolsig, e_n, extra

    # densidade de elétrons abaixo da qual a energia média tende à de campo local
    N_MIN_ENERGIA = 1.0e-2        # cm-3

    def energia_media(self, n, n_eps, e_n, t_gas=None, com_bruta=False):
        """Energia média (eV) por célula, presa à faixa da tabela na T_g da
        célula (com a tabela em T_g, o piso é ~(3/2) k T_g):
        (n_eps + n_min*eps_LFA)/(n_e + n_min). Com n_e >> n_min é n_eps/n_e; onde
        não há elétrons (n_e nulo ou levemente negativo pelo integrador) a razão
        não tem sentido e vale a energia de campo local. n_min fica bem acima do
        ruído das densidades no integrador (tolerância absoluta)."""
        n_min = self.N_MIN_ENERGIA
        tg = np.full(self.m, self.t_gas) if t_gas is None else np.asarray(t_gas, dtype=float)
        lfa = self.tab_e.energia_de_campo_local(e_n, tg)
        piso, teto = self.tab_e.limites(tg)
        ne = np.maximum(n[self.i_e], 0.0)
        # energia fora da faixa da tabela, contada antes de prender, onde há
        # elétrons bastantes para a razão ter sentido
        com_eletrons = ne > 1.0e3 * n_min
        if np.any(com_eletrons):
            bruta = n_eps[com_eletrons] / ne[com_eletrons]
            self.tab_e.fora += int(np.count_nonzero(bruta > teto[com_eletrons] * (1.0 + 1e-12)))
            self.tab_e.fora_abaixo += int(np.count_nonzero(bruta < piso[com_eletrons] * (1.0 - 1e-12)))
        # n_eps não passa de eps_max*n_e: sem elétrons, não há energia deles
        n_eps = np.clip(n_eps, 0.0, teto * ne)
        eps = (n_eps + n_min * lfa) / (ne + n_min)
        # com_bruta: também a energia antes de prender ao piso, para a
        # relaxação dos elétrons mais frios que o gás
        return (np.clip(eps, piso, teto), eps) if com_bruta else np.clip(eps, piso, teto)

    # ------------------------------------------------------------------ fluxos
    def sg(self, n, carga, mu_c, d_c, campo):
        """Fluxos de Scharfetter-Gummel nas faces internas para densidades n
        (linhas, células) de carga dada, com mobilidade e difusão das células."""
        d_int = self.d[1:-1]
        mu_f = media_harmonica(mu_c[:, :-1], mu_c[:, 1:])
        d_f = media_harmonica(d_c[:, :-1], d_c[:, 1:])
        w = carga[:, None] * mu_f * campo[None, 1:-1]
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            pe = np.where(d_f > 0.0, w * d_int / d_f, np.sign(w) * 1e300)
            gsg = (d_f / d_int) * (bernoulli(-pe) * n[:, :-1] - bernoulli(pe) * n[:, 1:])
        gup = np.maximum(w, 0.0) * n[:, :-1] + np.minimum(w, 0.0) * n[:, 1:]
        return np.where(d_f <= 0.0, gup, gsg)

    def fluxos(self, n, campo, mob, dif, temp):
        """Fluxos nas m+1 faces (cm-3 m/s, positivos para +x) e os fluxos que
        atingem cada parede, por espécie."""
        G = np.zeros((self.n_esp, self.m + 1))
        G[:, 1:-1] = self.sg(n, self.sinal, mob, dif, campo)
        # paredes
        v_term = np.sqrt(8.0 * KB * temp / (np.pi * self.massa[:, None]))
        out = np.zeros((self.n_esp, 2))
        for lado, (cel, face, sinal) in enumerate(((0, 0, -1.0), (self.m - 1, self.m, 1.0))):
            ce = self.carga * campo[face]                  # força por e: o freio usa |z| e E
            w_n = sinal * mob[:, cel] * self.sinal * campo[face]      # deriva na direção da parede
            meia = self.d[face]
            with np.errstate(over="ignore"):
                f = np.where((w_n < 0.0) & (temp[:, cel] > 0.0) & self.freio,
                             np.exp(-Q * np.abs(ce) * meia / (KB * temp[:, cel])), 1.0)
            out[:, lado] = n[:, cel] * (self.fator_parede * f * 0.5 * v_term[:, cel] + np.maximum(w_n, 0.0))
        gama_ions = self.gama * out[self.ions].sum(axis=0)
        G[:, 0] = -out[:, 0]
        G[:, -1] = out[:, 1]
        if self.gas_solto:
            # o gás de fundo não é absorvido: cada espécie dele recebe os átomos
            # que a parede absorveu (o fundo tem peso zero em atomos)
            retorno = self.atomos @ out
            for j, i in enumerate(self.i_fundos):
                G[i, 0] = retorno[j, 0]
                G[i, -1] = -retorno[j, 1]
        G[self.i_e, 0] += gama_ions[0]
        G[self.i_e, -1] -= gama_ions[1]
        return G, out, gama_ions

    def fluxos_energia(self, n, n_eps, campo, mob, temp, extra, gama_ions):
        """Fluxo de energia dos elétrons nas m+1 faces (eV cm-3 m/s, positivo
        para +x): Scharfetter-Gummel no interior; nas paredes, 5/2 kTe por
        elétron no fluxo térmico freado (fluxo de energia (5/6) v_term n_eps de
        Hagelaar et al., Phys. Rev. E 62, 1452, 2000), mu_eps*E*n_eps na deriva para a parede e eps_sec
        por elétron secundário que entra."""
        mu_e = extra["mu_eps"][None, :]
        d_e = extra["d_eps"][None, :]
        Ge = np.zeros(self.m + 1)
        Ge[1:-1] = self.sg(n_eps[None, :], np.array([-1.0]), mu_e, d_e, campo)[0]
        ie = self.i_e
        te = temp[ie]
        v_term = np.sqrt(8.0 * KB * te / (np.pi * ME))
        saida = np.zeros(2)
        for lado, (cel, face, sinal) in enumerate(((0, 0, -1.0), (self.m - 1, self.m, 1.0))):
            w_n = -sinal * mob[ie, cel] * campo[face]            # deriva dos elétrons para a parede
            w_eps = -sinal * extra["mu_eps"][cel] * campo[face]
            with np.errstate(over="ignore"):
                f = 1.0
                if w_n < 0.0 and self.freio:
                    f = np.exp(-Q * abs(campo[face]) * self.d[face] / (KB * te[cel]))
            saida[lado] = (n[ie, cel] * f * 0.5 * v_term[cel] * 2.5 * KB * te[cel] / Q
                           + max(w_eps, 0.0) * n_eps[cel])
        Ge[0] = -saida[0] + self.eps_sec * gama_ions[0]
        Ge[-1] = saida[1] - self.eps_sec * gama_ions[1]
        return Ge

    # ------------------------------------------------------------------- RHS
    def separa(self, y):
        """Densidades das espécies (sem a energia), cargas superficiais."""
        z = y[:-2].reshape(self.n_var, self.m)
        return z[:self.n_esp], y[-2], y[-1]

    def energia_do_estado(self, y):
        return y[:-2].reshape(self.n_var, self.m)[self.n_esp]

    def temperatura_do_estado(self, y):
        return y[:-2].reshape(self.n_var, self.m)[self.i_t]

    def densidade_neutra(self, n):
        """Densidade total de neutros por célula (cm-3): o ALL_NEUTRAL do
        ZDPlasKin, terceiro corpo das reações com ANY_NEUTRAL."""
        todas = self.densidades_completas(n)
        neutras = [s for s, nome in enumerate(self.q.especies) if self.q.carga[s] == 0 and nome.upper() != "E"]
        return todas[neutras].sum(axis=0)

    def densidades_completas(self, n):
        """Densidades de todas as espécies da química (cm-3), com o gás de
        fundo preso na sua fração de N; negativos do integrador viram zero."""
        todas = np.empty((len(self.q.especies), self.m))
        todas[self.especies] = np.maximum(n, 0.0)
        if not self.gas_solto:
            for i, fracao in self.fundos:
                todas[i] = fracao * self.n_gas
        return todas

    def campo(self, t, n, sigma_esq, sigma_dir):
        """Potencial nos nós e campo nas faces para as densidades n."""
        rho = Q * 1e6 * (self.carga @ n)
        return self.poisson(rho, sigma_esq, sigma_dir, self.tensao(t))

    def rhs(self, t, y):
        """dy/dt do estado completo, com o campo do Poisson no instante t."""
        n, s_esq, s_dir = self.separa(y)
        _, campo = self.campo(t, n, s_esq, s_dir)
        return self.rhs_campo(t, y, campo)

    def rhs_campo(self, t, y, campo):
        """Lado direito com o campo nas faces dado (o rhs o resolve antes; o
        jacobiano estruturado o perturba)."""
        n, s_esq, s_dir = self.separa(y)
        n_eps = self.energia_do_estado(y) if self.energia else None
        t_gas = self.temperatura_do_estado(y) if self.aquecimento else None
        campo_cel = self.campo_nas_celulas(campo)
        mob, dif, temp, k_bolsig, e_n, extra = self.coeficientes(campo_cel, n, n_eps, t_gas)
        G, out, gama_ions = self.fluxos(n, campo, mob, dif, temp)
        dn = -np.diff(G, axis=1) / self.h
        tg = t_gas if self.aquecimento else np.full(self.m, self.t_gas)
        k = self.q.coeficientes(k_bolsig, tg, temp[self.i_e], e_n,
                                self.densidade_neutra(n) if self.q.usa_any_neutral else None)
        if self.calor_reacao is not None:
            taxas = self.q.taxas(self.densidades_completas(n), k)
            extra["taxas"] = taxas
            fontes = self.q.estequiometria @ taxas
        else:
            fontes = self.q.fontes(self.densidades_completas(n), k)
        dn += fontes[self.especies]
        partes = [dn.ravel()]
        if self.energia:
            partes.append(self.d_energia(n, n_eps, campo, G, mob, temp, extra, gama_ions))
        if self.aquecimento:
            partes.append(self.d_temperatura(n, t_gas, campo_cel, mob, extra))
        # cargas superficiais: as partículas que chegam e os elétrons emitidos
        d_sigma = Q * 1e6 * (self.carga @ out + gama_ions)
        return np.concatenate(partes + [d_sigma])

    def d_temperatura(self, n, t_gas, campo_cel, mob, extra):
        """dT/dt (K/s): condução com kappa(T) (média harmônica nas faces, parede
        a t_parede a meia célula), perda elástica dos elétrons e Joule dos íons,
        divididos por (3/2) N k_B com N as partículas pesadas."""
        kap = self.condutividade(t_gas)
        k_par = self.condutividade(np.array(self.t_parede))
        F = np.empty(self.m + 1)                         # W/m2, positivo para +x
        F[1:-1] = -media_harmonica(kap[:-1], kap[1:]) * np.diff(t_gas) / self.d[1:-1]
        F[0] = -media_harmonica(k_par, kap[0]) * (t_gas[0] - self.t_parede) / self.d[0]
        F[-1] = -media_harmonica(kap[-1], k_par) * (self.t_parede - t_gas[-1]) / self.d[-1]
        ne = np.maximum(n[self.i_e], 0.0)
        q_el = Q * extra["pel"] * extra["n"] * ne * 1e6                 # W/m3
        q_j = self.joule_ions(n, mob, campo_cel)
        fonte = -np.diff(F) / self.h + q_el + q_j
        if self.calor_reacao is not None:
            # calor das reações (entalpias do gas.json) e, com a equação de energia,
            # a energia média que os elétrons consumidos levam (d_energia a retira)
            q_reac = self.calor_reacao @ extra["taxas"]
            if self.energia:
                q_reac = q_reac - extra["eps"] * (self._eletrons_perdidos @ extra["taxas"])
            fonte = fonte + Q * 1e6 * q_reac
        if self.tab_calor is not None:
            # P/N dos processos rápidos na energia média (com --energia) ou na de
            # campo local da própria tabela, no E/N e na T_g da célula
            tg = extra["tg"]
            eps = extra["eps"] if self.energia else self.tab_calor.energia_de_campo_local(extra["e_n"], tg)
            p_rap = self.tab_calor(eps, tg)[:, 1]
            fonte = fonte + Q * 1e6 * p_rap * extra["n"] * ne
        return fonte / (self.cv_por_kb * KB * self.n_pesadas(n) * 1e6)

    def joule_ions(self, n, mob, campo_cel):
        """Aquecimento Joule dos íons (W/m3), |z| e n mu E^2, positivos e negativos."""
        ions = self.ions + self.ions_negativos
        return sum(abs(self.carga[k]) * Q * np.maximum(n[k], 0.0) * 1e6 * mob[k] * campo_cel ** 2 for k in ions)

    def ganho_de_energia(self, G, campo):
        """Trabalho do campo sobre os elétrons por célula (eV cm-3/s), -Gamma_e*E
        médio nas duas faces da célula. Nas células das paredes vale só o da
        face interna: o fluxo de absorção na face da parede é dominado pela
        agitação térmica (v_term/2 ~ 1e6 m/s), que o campo não realiza, e
        aquecia a célula várias vezes mais que o volume."""
        trabalho = -G[self.i_e] * campo
        ganho = 0.5 * (trabalho[:-1] + trabalho[1:])
        ganho[0] = trabalho[1]
        ganho[-1] = trabalho[-2]
        return ganho

    def d_energia(self, n, n_eps, campo, G, mob, temp, extra, gama_ions):
        """d(n_eps)/dt (eV cm-3/s): transporte, ganho pelo campo com os fluxos de
        elétrons das duas faces da célula, perda elástica e inelástica tabelada
        (P_el + P_inel), troca de energia nas colisões com Ar*, energia média
        levada por elétron recombinado e o ganho opcional do terceiro elétron na
        recombinação a três corpos."""
        Ge = self.fluxos_energia(n, n_eps, campo, mob, temp, extra, gama_ions)
        ganho = self.ganho_de_energia(G, campo)
        perda = np.maximum(n[self.i_e], 0.0) * extra["n"] * extra["perda"]
        # reações: superelásticas e ionização em etapas com o Ar* dão ou tiram a
        # energia do limiar; cada elétron perdido na recombinação leva a energia
        # média (os elétrons novos entram sem energia: a ionização já paga o
        # limiar em P_inel e o resto fica com os elétrons)
        k = self.q.coeficientes(extra["k_bolsig"], extra["tg"], temp[self.i_e], extra["e_n"],
                                self.densidade_neutra(n) if self.q.usa_any_neutral else None)
        taxas = self.q.taxas(self.densidades_completas(n), k)
        perdidos = np.minimum(self.q.estequiometria[self.q.indice["E"]], 0.0)
        outras = self.energia_reacao @ taxas + extra["eps"] * (perdidos @ taxas)
        return -np.diff(Ge) / self.h + ganho - perda + outras

    def fora_da_tabela(self, y):
        """Células de um estado com energia média acima e abaixo da faixa da
        tabela, onde há elétrons bastantes (n_e > 10 cm-3), para o log."""
        n = self.separa(y)[0]
        ne = n[self.i_e]
        com = ne > 1.0e3 * self.N_MIN_ENERGIA
        bruta = self.energia_do_estado(y)[com] / ne[com]
        tg = self.temperatura_do_estado(y) if self.aquecimento else np.full(self.m, self.t_gas)
        piso, teto = self.tab_e.limites(tg)
        return (int(np.count_nonzero(bruta > teto[com] * (1.0 + 1e-12))),
                int(np.count_nonzero(bruta < piso[com] * (1.0 - 1e-12))))

    def frequencia_de_colisao(self, n, extra):
        """Frequência de colisão dos elétrons (1/s) por célula, como o Fortran:
        (nu/N) da tabela vezes a soma de todas as densidades da célula."""
        return extra["nu"] * (self.n_pesadas(n) + np.maximum(n[self.i_e], 0.0))

    def celula_muito_ionizada(self, y):
        """Primeira célula com n_e/N acima de 1e-3 com o gás preso, onde a
        hipótese do gás preso deixa de valer (aviso do Fortran); None se não há."""
        if self.gas_solto:
            return None
        acima = np.nonzero(self.separa(y)[0][self.i_e] > 1e-3 * self.n_gas)[0]
        return int(acima[0]) if len(acima) else None

    T_ALTA = 1000.0     # K: acima disso o kappa extrapolado deixa de valer

    def celula_quente(self, y):
        """Primeira célula com o gás acima de T_ALTA (com aquecimento); None se não há."""
        if not self.aquecimento:
            return None
        acima = np.nonzero(self.temperatura_do_estado(y) > self.T_ALTA)[0]
        return int(acima[0]) if len(acima) else None

    def estado_inicial(self, n_e0=1.0, neutro=False):
        """Como o Fortran: só elétrons a n_e0 cm-3, superfícies descarregadas;
        com neutro, o íon inicial do gas.json também a n_e0
        (plasma inicial quase neutro); com a
        equação de energia, a energia média de campo nulo."""
        n = np.zeros((self.n_var, self.m))
        n[self.i_e] = n_e0
        if neutro:
            n[self.ion_inicial] = n_e0
        if self.energia:
            n[self.n_esp] = n_e0 * self.tab_e.limites(np.array([self.t_gas]))[0][0]
        if self.aquecimento:
            n[self.i_t] = self.t_gas
        if self.gas_solto:
            for (_, fracao), i in zip(self.fundos, self.i_fundos):
                n[i] = fracao * self.n_gas
        return np.concatenate([n.ravel(), [0.0, 0.0]])
