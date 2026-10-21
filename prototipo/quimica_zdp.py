# SPDX-License-Identifier: GPL-3.0-or-later
"""Química lida do zdplaskin_m.F90 que o pré-processador do ZDPlasKin gera a
partir do kinet.inp, para qualquer mistura: espécies, cargas, expressões das
taxas, ordens dos reagentes e estequiometria. Nada é escrito à mão.

As reações de impacto eletrônico (rrt(i) = bolsig_rates(bolsig_pointer(k)))
usam a coluna k das taxas da tabela em E/N; as demais são as expressões do
ZDPlasKin_reac_rates, avaliadas com TGAS e TE.

A rotina de taxas vira um programa avaliado em ordem, célula a célula: as
atribuições a variáveis locais e a rrt(i), inclusive dentro de IF de uma linha
ou em bloco (ELSE IF, ELSE), que viram np.where com a condição. A divisão entre
inteiros segue o Fortran (1/2 vale 0), com aviso.
"""
import ast
import functools
import json
import pathlib
import re

import numpy as np


def _bloco(texto, inicio, fim):
    i = texto.index(inicio)
    return texto[i:texto.index(fim, i)]


def _lista_data(texto, nome):
    """Valores de data NOME(a:b) /.../; com listas longas o pré-processador
    divide em vários blocos (1:54, 55:108, ...), juntados na ordem de a."""
    blocos = re.findall(r"data\s+" + re.escape(nome) + r"\((\d+):\w+\)\s*&\s*/(.*?)/", texto, re.S)
    valores = []
    for _, corpo in sorted(blocos, key=lambda b: int(b[0])):
        corpo = corpo.replace("&", " ")
        if '"' in corpo:                 # nomes: podem ter vírgula, como N2(A3,V0-4)
            valores += [x.strip() for x in re.findall(r'"([^"]*)"', corpo)]
        else:
            valores += [x.strip() for x in corpo.split(",")]
    return valores


class ErroQuimica(ValueError):
    """zdplaskin_m.F90 com construção que o leitor não sabe avaliar."""


def _le_limiares(bolsigdb):
    """Energia (eV) dada ao elétron em cada processo do bolsigdb.dat, por (alvo,
    produto): -limiar na excitação e na ionização, +limiar no sentido inverso de
    um processo X <-> Y, zero na captura (o formato do LXCat não traz limiar)."""
    processos = {}
    linhas = pathlib.Path(bolsigdb).read_text().splitlines()
    for i, l in enumerate(linhas):
        if l.strip() in ("EXCITATION", "IONIZATION", "ATTACHMENT"):
            proc = linhas[i + 1].replace(" ", "").upper()
            if l.strip() == "ATTACHMENT":
                limiar = 0.0
            else:
                try:
                    limiar = float(linhas[i + 2].split("/")[0].split()[0])
                except (ValueError, IndexError):
                    raise ErroQuimica(f"{bolsigdb}: processo {proc} sem a linha do limiar (eV)") from None
            if "<->" in proc:
                a, b = proc.split("<->")
                processos[(a, b)] = -limiar
                processos[(b, a)] = limiar
            else:
                a, b = proc.split("->")
                processos[(a, b)] = -limiar
    return processos


def _maximo(*a):
    return functools.reduce(np.maximum, a)


def _minimo(*a):
    return functools.reduce(np.minimum, a)


# funções do Fortran aceitas nas expressões de taxa, e as variáveis que o
# ZDPlasKin_reac_rates recebe: TGAS e TE em K, EN em Td
FUNCOES = {"exp": np.exp, "sqrt": np.sqrt, "log": np.log, "log10": np.log10, "abs": np.abs,
           "max": _maximo, "min": _minimo, "dexp": np.exp, "dsqrt": np.sqrt, "dlog": np.log}
ENTRADAS = ("tgas", "te", "en", "any_neutral")


# literal numérico do Fortran: inteiro, real simples (com ponto ou expoente E)
# ou real duplo (expoente D)
_LITERAL = re.compile(r"(?<![\w.])(\d+\.\d*|\.\d+|\d+)(?:([eEdD])([+-]?\d+))?(?![\w.])")


def _literal_python(m):
    """Literal do Fortran em Python com o mesmo valor: o real sem expoente D é
    de precisão simples no Fortran (0.39 vale 0,38999998569...), então entra
    arredondado para float32; o de expoente D é duplo; o inteiro fica inteiro."""
    mantissa, letra, expoente = m.group(1), m.group(2), m.group(3)
    if letra is None and "." not in mantissa:
        return mantissa
    texto = mantissa + ("e" + expoente if letra else "")
    if letra in ("d", "D"):
        return repr(float(texto))
    # marcado como real simples: as contas só entre eles também são simples
    return f"_s({float(np.float32(float(texto)))!r})"


def _expressao_python(expr):
    """Expressão Fortran do ZDPlasKin_reac_rates em Python com numpy: literais
    com o valor que o Fortran usa (real simples, real duplo, inteiro) e nomes
    em minúsculas (o Fortran não distingue)."""
    e = _LITERAL.sub(_literal_python, expr)
    return re.sub(r"(?<!\d)[A-Za-z_][A-Za-z_0-9]*", lambda m: m.group(0).lower(), e)



# operadores relacionais e lógicos do Fortran, já em minúsculas
_OPERADORES = ((".gt.", " > "), (".lt.", " < "), (".ge.", " >= "), (".le.", " <= "), (".eq.", " == "),
               (".ne.", " != "), ("/=", " != "), (".and.", " and "), (".or.", " or "), (".not.", " not "),
               (".true.", " True "), (".false.", " False "))

# ZDPlasKin_cfg com sentido na rotina de taxas: T_g, E/N e T_e
_CFG = {1: "tgas", 3: "en", 4: "te"}


def _divisao_fortran(a, b):
    """Divisão inteira do Fortran: truncada para zero."""
    q = abs(a) // abs(b)
    return q if (a >= 0) == (b >= 0) else -q


class _Fortran(ast.NodeTransformer):
    """Semântica do Fortran na árvore da expressão: and, or e not elemento a
    elemento (numpy) e os tipos dos números. Conta entre inteiros é inteira (a
    divisão truncada: 1/2 vale 0); conta entre reais simples, ou entre simples e
    inteiro, é simples e sai arredondada para float32 (1.0/3.0 vale
    0,33333334); com um operando duplo (um literal com D ou qualquer variável,
    todas double precision na rotina de taxas) a conta é dupla. A precedência
    do Python em *, / e ** é a do Fortran (x*3/2 é (x*3)/2, real)."""

    def __init__(self):
        self.divisoes_inteiras = 0

    @staticmethod
    def _tipo(no):
        """'i' inteiro, 's' real simples ou 'd' duplo."""
        if isinstance(no, ast.Constant):
            return "i" if isinstance(no.value, int) and not isinstance(no.value, bool) else "d"
        if isinstance(no, ast.UnaryOp) and isinstance(no.op, (ast.USub, ast.UAdd)):
            return _Fortran._tipo(no.operand)
        if isinstance(no, ast.Call) and isinstance(no.func, ast.Name):
            return {"_divisao_fortran": "i", "_s": "s", "_simples": "s"}.get(no.func.id, "d")
        if isinstance(no, ast.BinOp):
            tipos = {_Fortran._tipo(no.left), _Fortran._tipo(no.right)}
            return "d" if "d" in tipos else ("s" if "s" in tipos else "i")
        return "d"

    def visit_BinOp(self, no):
        self.generic_visit(no)
        esq, dir_ = self._tipo(no.left), self._tipo(no.right)
        if isinstance(no.op, ast.Div) and esq == dir_ == "i":
            self.divisoes_inteiras += 1
            return ast.Call(ast.Name("_divisao_fortran", ast.Load()), [no.left, no.right], [])
        if "d" not in (esq, dir_) and "s" in (esq, dir_):
            return ast.Call(ast.Name("_simples", ast.Load()), [no], [])
        return no

    def visit_BoolOp(self, no):
        self.generic_visit(no)
        nome = "_e" if isinstance(no.op, ast.And) else "_ou"
        return ast.Call(ast.Name(nome, ast.Load()), no.values, [])

    def visit_UnaryOp(self, no):
        self.generic_visit(no)
        if isinstance(no.op, ast.Not):
            return ast.Call(ast.Name("_nao", ast.Load()), [no.operand], [])
        return no


AUXILIARES = {"_e": lambda *a: functools.reduce(np.logical_and, a),
              "_ou": lambda *a: functools.reduce(np.logical_or, a),
              "_nao": np.logical_not, "_onde": np.where, "_nada": np.nan,
              "_s": lambda x: x, "_simples": lambda x: float(np.float32(x))}


def _fim_da_condicao(linha, inicio):
    """Índice do parêntese que fecha a condição de um IF; erro claro se não fecha."""
    try:
        return _fecha_parenteses(linha, inicio)
    except ValueError:
        raise ErroQuimica(f"IF sem fechar o parêntese: {linha}") from None


def _fecha_parenteses(texto, inicio):
    """Índice do parêntese que fecha o aberto em texto[inicio]."""
    nivel = 0
    for i in range(inicio, len(texto)):
        nivel += (texto[i] == "(") - (texto[i] == ")")
        if nivel == 0:
            return i
    raise ValueError("parênteses sem fechar")


def _linhas_logicas(bloco):
    """Linhas do bloco com as continuações (& no fim) juntadas."""
    saida, atual = [], ""
    for linha in bloco.splitlines():
        codigo = linha.split("!")[0].rstrip()
        parte = codigo.strip()
        if atual:
            parte = parte.lstrip("&").strip()
        if parte.endswith("&"):
            atual += parte[:-1] + " "
            continue
        saida.append(atual + parte)
        atual = ""
    if atual:
        saida.append(atual)
    return saida


def _separa_no_nivel_zero(texto):
    """Partes de texto separadas pelas vírgulas que não estão dentro de parênteses."""
    partes, nivel, atual = [], 0, ""
    for c in texto:
        if c == "," and nivel == 0:
            partes.append(atual)
            atual = ""
            continue
        nivel += (c == "(") - (c == ")")
        atual += c
    return partes + [atual]


class Quimica:
    def __init__(self, arquivo):
        texto = pathlib.Path(arquivo).read_text()
        self.especies = _lista_data(texto, "species_name")
        self.carga = np.array([int(x) for x in _lista_data(texto, "species_charge")], dtype=float)
        self.reacoes = _lista_data(texto, "reaction_sign")
        ne, nr = len(self.especies), len(self.reacoes)
        # coeficientes: coluna da tabela (k da BOLSIG+) ou expressão
        taxas = _bloco(texto, "subroutine ZDPlasKin_reac_rates", "end subroutine ZDPlasKin_reac_rates")
        self.coluna_bolsig = [None] * nr
        self.avisos = []
        self._le_programa(_linhas_logicas(taxas)[1:])
        # ordens dos reagentes: rrt(i) = rrt(i) * density(a)**p * density(b) ...
        fex = _bloco(texto, "subroutine ZDPlasKin_fex", "end subroutine ZDPlasKin_fex")
        self.ordem = np.zeros((ne, nr))
        for m in re.finditer(r"rrt\((\d+)\)\s*=\s*rrt\(\d+\)\s*\*(.+)", fex):
            i = int(m.group(1)) - 1
            for d in re.finditer(r"density\((\d+)\)(\*\*(\d+))?", m.group(2)):
                self.ordem[int(d.group(1)) - 1, i] += int(d.group(3) or 1)
        # estequiometria: reac_source_local(s,r) = +- reac_rate_local(r) [* f]
        matriz = _bloco(texto, "subroutine ZDPlasKin_reac_source_matrix",
                        "end subroutine ZDPlasKin_reac_source_matrix")
        self.estequiometria = np.zeros((ne, nr))
        for m in re.finditer(r"reac_source_local\((\d+),(\d+)\)\s*=\s*([+-])\s*reac_rate_local\(\d+\)"
                             r"(\s*\*\s*([\d.]+)[dD]0)?", matriz):
            f = float(m.group(5)) if m.group(5) else 1.0
            self.estequiometria[int(m.group(1)) - 1, int(m.group(2)) - 1] = (-f if m.group(3) == "-" else f)
        self.indice = {nome.upper(): k for k, nome in enumerate(self.especies)}
        # reações do BOLSIG+ com dois ou mais elétrons reagentes: a taxa do
        # BOLSIG+ é de dois corpos (cm3/s) e não serve de coeficiente de três corpos
        ie = self.indice["E"]
        self.bolsig_tres_corpos = [self.reacoes[i] for i in range(nr)
                                   if self.coluna_bolsig[i] is not None and self.ordem[ie, i] >= 2]

    def energia_eletronica(self, bolsigdb, fundo):
        """Energia (eV) que cada reação de impacto eletrônico dá ao elétron, para as
        reações com alvo diferente do gás de fundo, cujas perdas a tabela de Ar
        puro não contém: -limiar para excitação e ionização, +limiar para a
        superelástica (processo X <-> Y do bolsigdb.dat percorrido de Y para X).
        Zero nas demais reações (as do gás de fundo já estão na potência tabelada).
        fundo: nome da espécie do fundo, ou conjunto de nomes numa mistura.
        bolsigdb None: os limiares guardados no quimica.json."""
        fundos = {fundo.upper()} if isinstance(fundo, str) else {f.upper() for f in fundo}
        processos = (getattr(self, "_limiares", None) or {}) if bolsigdb is None else _le_limiares(bolsigdb)
        energia = np.zeros(len(self.reacoes))
        for i, r in enumerate(self.reacoes):
            if self.coluna_bolsig[i] is None:
                continue
            a, b = r.split(":", 1)[1].upper().split("->")
            if a not in fundos:
                if (a, b) not in processos:
                    onde = "no quimica.json" if bolsigdb is None else f"no {bolsigdb}"
                    raise ErroQuimica(f"reação {i + 1} ({r}): sem o limiar de {a} -> {b} {onde}")
                energia[i] = processos[(a, b)]
        return energia

    def salva(self, caminho, bolsigdb=None):
        """Grava a química em JSON (quimica.json): espécies, cargas, reações,
        ordens, estequiometria e o programa das taxas como texto Python, mais os
        limiares do bolsigdb.dat das reações do BOLSIG+. O arquivo não tem nada
        do código do ZDPlasKin: é o kinet.inp traduzido."""
        programa = []
        for (alvo, codigo, onde), texto in zip(self._programa, self._textos):
            p = {"alvo": alvo, "onde": onde}
            if texto is None:
                p["coluna_bolsig"] = codigo
            else:
                p["texto"] = texto
            programa.append(p)
        limiares = []
        if bolsigdb is not None:
            todos = _le_limiares(bolsigdb)
            for i, r in enumerate(self.reacoes):
                if self.coluna_bolsig[i] is not None:
                    a, b = r.split(":", 1)[1].upper().split("->")
                    if (a, b) in todos:
                        limiares.append([a, b, todos[(a, b)]])
        dados = {"formato": "quimica.json 1", "especies": self.especies, "carga": self.carga.tolist(),
                 "reacoes": self.reacoes, "coluna_bolsig": self.coluna_bolsig, "ordem": self.ordem.tolist(),
                 "estequiometria": self.estequiometria.tolist(), "programa": programa,
                 "usa_en": self.usa_en, "usa_any_neutral": self.usa_any_neutral, "avisos": self.avisos,
                 "limiares_bolsig": limiares}
        pathlib.Path(caminho).write_text(json.dumps(dados, ensure_ascii=False, indent=1) + "\n")

    @classmethod
    def de_json(cls, caminho):
        """Química gravada por salva, sem o zdplaskin_m.F90."""
        d = json.loads(pathlib.Path(caminho).read_text())
        if d.get("formato") != "quimica.json 1":
            raise ErroQuimica(f"{caminho}: formato {d.get('formato')!r} desconhecido (esperado 'quimica.json 1')")
        q = cls.__new__(cls)
        q.especies, q.reacoes = d["especies"], d["reacoes"]
        q.carga = np.array(d["carga"], dtype=float)
        q.coluna_bolsig = d["coluna_bolsig"]
        q.ordem, q.estequiometria = np.array(d["ordem"]), np.array(d["estequiometria"])
        q.usa_en, q.usa_any_neutral, q.avisos = d["usa_en"], d["usa_any_neutral"], d["avisos"]
        q._programa, q._textos = [], []
        for p in d["programa"]:
            if "texto" in p:
                q._programa.append((p["alvo"], compile(p["texto"], p["onde"], "eval"), p["onde"]))
                q._textos.append(p["texto"])
            else:
                q._programa.append((p["alvo"], p["coluna_bolsig"], p["onde"]))
                q._textos.append(None)
        q._indisponiveis = {}
        q._limiares = {(a, b): v for a, b, v in d["limiares_bolsig"]}
        q.do_json = True
        q.indice = {nome.upper(): k for k, nome in enumerate(q.especies)}
        ie = q.indice["E"]
        q.bolsig_tres_corpos = [q.reacoes[i] for i in range(len(q.reacoes))
                                if q.coluna_bolsig[i] is not None and q.ordem[ie, i] >= 2]
        return q

    def _le_programa(self, linhas):
        """Programa da rotina de taxas: (alvo, código) em ordem, com alvo o
        nome da variável local ou o índice da reação; dentro de IF o código é
        np.where(condição, expressão, valor anterior)."""
        self._programa = []
        self._textos = []                # o texto Python de cada passo (None: coluna do BOLSIG+), para exportar
        conhecidos = set(FUNCOES) | set(ENTRADAS)
        definidas = set()
        rrt_definidas = set()
        pilha = []                       # por IF aberto: [condição do ramo, algum ramo anterior]
        expressoes = set()               # reações com taxa por expressão
        # variáveis que o leitor não sabe calcular (ZDPlasKin_cfg fora de 1, 3 e
        # 4, como o De que o pré-processador emite): erro só se alguém as usa
        self._indisponiveis = {}
        padrao_ent = re.compile(r"(?i)^(tgas|en|te)\s*=\s*zdplasKin_cfg\(\d+\)$")

        def condicao():
            ramos = [c for c, _ in pilha]
            return ramos[0] if len(ramos) == 1 else f"_e({', '.join(ramos)})" if ramos else None

        def atribui(alvo, direita, onde):
            try:
                src = self._traduz(direita, onde, conhecidos | {"rrt_k", "k_bolsig"})
            except ErroQuimica as erro:
                if isinstance(alvo, int) or pilha or "ZDPLASKIN_CFG" not in str(erro):
                    raise
                self._indisponiveis[alvo] = str(erro)
                return
            if isinstance(alvo, int):
                anterior = f"rrt_k[{alvo}]" if alvo in rrt_definidas else "_nada"
                rrt_definidas.add(alvo)
            else:
                anterior = alvo if alvo in definidas else "_nada"
                definidas.add(alvo)
                conhecidos.add(alvo)
            c = condicao()
            if c is not None:
                src = f"_onde({c}, ({src}), {anterior})"
            if isinstance(alvo, int):
                expressoes.add(alvo)
            self._programa.append((alvo, compile(src, onde, "eval"), onde))
            self._textos.append(src)

        def instrucao(linha):
            minus = linha.lower()
            if minus.startswith("call zdplaskin_get_density_total") and "any_neutral" in minus:
                return          # ANY_NEUTRAL: entrada de coeficientes, a densidade de neutros
            if minus.startswith("call ") and not minus.startswith("call zdplaskin_bolsig_rates"):
                raise ErroQuimica(f"linha do ZDPlasKin_reac_rates com call, que o leitor não executa: {linha}")
            if "::" in linha:
                # declaração; com valor (parameter), vira variável local; várias
                # declarações separadas por vírgula fora de parênteses
                for item in _separa_no_nivel_zero(linha.split("::", 1)[1]):
                    if "=" in item:
                        nome, valor = (x.strip() for x in item.split("=", 1))
                        atribui(nome.lower(), valor, f"variável {nome}")
                return
            if minus == "return" and pilha:
                raise ErroQuimica(f"RETURN dentro de IF no ZDPlasKin_reac_rates, que o leitor não segue: {linha}")
            if (not linha or minus.startswith(("use ", "implicit", "call ", "where", "return", "lreaction_block"))
                    or padrao_ent.match(linha)):
                return
            m = re.match(r"(?i)if\s*\(", linha)
            if m:
                fim = _fim_da_condicao(linha, m.end() - 1)
                cond = self._traduz(linha[m.end():fim], f"condição em {linha}", conhecidos)
                resto = linha[fim + 1:].strip()
                if resto.lower() == "then":
                    pilha.append([cond, cond])
                else:
                    pilha.append([cond, cond])
                    instrucao(resto)
                    pilha.pop()
                return
            m = re.match(r"(?i)else\s*if\s*\(", linha)
            if m:
                if not pilha:
                    raise ErroQuimica(f"ELSE IF sem IF: {linha}")
                fim = _fim_da_condicao(linha, m.end() - 1)
                cond = self._traduz(linha[m.end():fim], f"condição em {linha}", conhecidos)
                anterior = pilha[-1][1]
                pilha[-1] = [f"_e(_nao({anterior}), {cond})", f"_ou({anterior}, {cond})"]
                return
            if minus == "else":
                if not pilha:
                    raise ErroQuimica("ELSE sem IF")
                pilha[-1][0] = f"_nao({pilha[-1][1]})"
                return
            if minus.replace(" ", "") == "endif":
                if not pilha:
                    raise ErroQuimica("END IF sem IF")
                pilha.pop()
                return
            m = re.match(r"(?i)rrt\((\d+)\)\s*=\s*(.+)", linha)
            if m:
                i = int(m.group(1)) - 1
                direita = m.group(2).strip()
                b = re.match(r"bolsig_rates\(bolsig_pointer\((\d+)\)\)$", direita)
                if b:
                    if pilha:
                        raise ErroQuimica(f"reação {i + 1}: taxa do BOLSIG+ dentro de IF, que o leitor não "
                                          f"separa por célula na energia e no calor: {linha}")
                    self.coluna_bolsig[i] = int(b.group(1)) - 1
                    self._programa.append((i, int(b.group(1)) - 1, f"reação {i + 1}"))
                    self._textos.append(None)
                    rrt_definidas.add(i)
                    return
                # rrt(j): o coeficiente de uma reação anterior (antes das densidades)
                for j in re.findall(r"(?i)rrt\((\d+)\)", direita):
                    if int(j) - 1 >= i:
                        raise ErroQuimica(f"reação {i + 1}: usa a taxa da reação {int(j)}, posterior a ela: {linha}")
                direita = re.sub(r"(?i)rrt\((\d+)\)", lambda r: f"rrt_k[{int(r.group(1)) - 1}]", direita)
                atribui(i, direita, f"reação {i + 1}")
                return
            m = re.match(r"([A-Za-z_][A-Za-z_0-9]*)\s*=\s*(.+)$", linha)
            if not m:
                raise ErroQuimica(f"linha do ZDPlasKin_reac_rates que o leitor não sabe avaliar: {linha}")
            atribui(m.group(1).lower(), m.group(2), f"variável {m.group(1)}")

        for linha in linhas:
            instrucao(linha)
        if pilha:
            raise ErroQuimica("IF sem end if no ZDPlasKin_reac_rates")
        mistas = sorted(i + 1 for i in expressoes if self.coluna_bolsig[i] is not None)
        if mistas:
            raise ErroQuimica(f"reações {', '.join(map(str, mistas))}: taxa do BOLSIG+ redefinida por expressão; "
                              "o leitor trata a reação inteira como do BOLSIG+ na energia e no calor")
        codigos = [c for _, c, _ in self._programa if not isinstance(c, int)]
        self.usa_en = any("en" in c.co_names for c in codigos)
        self.usa_any_neutral = any("any_neutral" in c.co_names for c in codigos)

    def _traduz(self, texto, onde, conhecidos):
        """Expressão (ou condição) Fortran em código Python válido, com os nomes
        conferidos; a divisão entre inteiros vira a do Fortran, com aviso."""
        texto = texto.strip()
        e = _expressao_python(texto)
        for fortran, python in _OPERADORES:
            e = e.replace(fortran, python)
        e = re.sub(r"zdplaskin_cfg\((\d+)\)", lambda m: _CFG.get(int(m.group(1)), m.group(0).upper()), e).strip()
        if "ZDPLASKIN_CFG" in e:
            usados = ", ".join(sorted(set(re.findall(r"ZDPLASKIN_CFG\(\d+\)", e))))
            raise ErroQuimica(f"{onde}: {usados} em "
                              f"{texto}; na rotina de taxas o leitor conhece ZDPlasKin_cfg(1), (3) e (4), "
                              "T_g, E/N e T_e")
        fortran = _Fortran()
        try:
            arvore = fortran.visit(ast.parse(e, mode="eval"))
        except SyntaxError as erro:
            raise ErroQuimica(f"{onde}: expressão que o leitor não sabe avaliar: {texto} ({erro.msg})") from None
        if fortran.divisoes_inteiras:
            self.avisos.append(f"{onde}: divisão entre inteiros em {texto}, feita como no Fortran "
                               "(truncada: 1/2 vale 0); escreva 1.0/2.0 se a intenção era real")
        codigo = compile(ast.fix_missing_locations(arvore), onde, "eval")
        desconhecidos = sorted(set(codigo.co_names) - conhecidos - set(AUXILIARES) - {"_divisao_fortran"})
        for nome in desconhecidos:
            if nome in getattr(self, "_indisponiveis", {}):
                raise ErroQuimica(f"{onde} usa {nome.upper()}, que o leitor não calcula: {self._indisponiveis[nome]}")
        if desconhecidos:
            raise ErroQuimica(f"{onde}: nome desconhecido {', '.join(n.upper() for n in desconhecidos)} em {texto}")
        return ast.unparse(arvore)

    def coeficientes(self, k_bolsig, tgas, te, en=None, any_neutral=None):
        """Coeficientes das reações por célula. k_bolsig: (células, colunas) da
        tabela; tgas e te em K, en (E/N local, Td) e any_neutral (densidade total
        de neutros, cm-3, o ALL_NEUTRAL do ZDPlasKin) por célula. Devolve
        (reações, células)."""
        if en is None and self.usa_en:
            raise ErroQuimica("a química usa EN (E/N local): passe en a coeficientes")
        if any_neutral is None and self.usa_any_neutral:
            raise ErroQuimica("a química usa ANY_NEUTRAL (densidade de neutros): passe any_neutral a coeficientes")
        nomes = dict(FUNCOES, **AUXILIARES, _divisao_fortran=_divisao_fortran,
                     tgas=tgas, te=te, en=en, any_neutral=any_neutral, k_bolsig=k_bolsig)
        k = np.full((len(self.reacoes), np.size(te)), np.nan)
        nomes["rrt_k"] = k
        for alvo, codigo, _ in self._programa:
            if isinstance(codigo, int):
                k[alvo] = k_bolsig[:, codigo]
            elif isinstance(alvo, int):
                k[alvo] = eval(codigo, {"__builtins__": {}}, nomes)
            else:
                nomes[alvo] = eval(codigo, {"__builtins__": {}}, nomes)
        entradas = [tgas, te, k_bolsig] + [x for x in (en, any_neutral) if x is not None]
        if np.isnan(k).any() and all(np.isfinite(x).all() for x in entradas):
            # com as entradas finitas, NaN é valor indefinido: variável ou taxa
            # atribuída só num ramo de IF (um passo ruim do integrador, com
            # entrada NaN, segue para ele rejeitar)
            r = int(np.nonzero(np.isnan(k).any(axis=1))[0][0])
            raise ErroQuimica(f"reação {r + 1}: coeficiente sem valor em parte das células (variável ou taxa "
                              "atribuída só num ramo de IF, sem ELSE nem valor antes)")
        return k

    def taxas(self, dens, k):
        """Taxa de cada reação (reações, células); dens em cm-3 (espécies, células)."""
        produto = np.ones_like(k)
        for s in range(len(self.especies)):
            for i in np.nonzero(self.ordem[s])[0]:
                produto[i] *= dens[s] ** self.ordem[s, i]
        return k * produto

    def fontes(self, dens, k):
        """Termos-fonte (espécies, células) em cm-3/s."""
        return self.estequiometria @ self.taxas(dens, k)
