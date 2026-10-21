# SPDX-License-Identifier: GPL-3.0-or-later
"""Chemistry read from the zdplaskin_m.F90 that the ZDPlasKin preprocessor
generates from kinet.inp, for any mixture: species, charges, rate expressions,
reactant orders and stoichiometry. Nothing is written by hand.

Electron impact reactions (rrt(i) = bolsig_rates(bolsig_pointer(k))) use column k
of the rate coefficients of the electron tables; the others are the expressions of
ZDPlasKin_reac_rates, evaluated with TGAS and TE.

The rate routine becomes a program evaluated in order, for all cells at once: the
assignments to local variables and to rrt(i), also inside one-line or block IF
statements (ELSE IF, ELSE), which become np.where with the condition. Integer
division follows Fortran (1/2 is 0), with a warning.
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
    """Values of data NAME(a:b) /.../; with long lists the preprocessor splits
    them into several blocks (1:54, 55:108, ...), joined in the order of a."""
    blocos = re.findall(r"data\s+" + re.escape(nome) + r"\((\d+):\w+\)\s*&\s*/(.*?)/", texto, re.S)
    valores = []
    for _, corpo in sorted(blocos, key=lambda b: int(b[0])):
        corpo = corpo.replace("&", " ")
        if '"' in corpo:                 # names: may contain commas, as N2(A3,V0-4)
            valores += [x.strip() for x in re.findall(r'"([^"]*)"', corpo)]
        else:
            valores += [x.strip() for x in corpo.split(",")]
    return valores


class ErroQuimica(ValueError):
    """zdplaskin_m.F90 with a construct that the reader cannot evaluate."""


def _le_limiares(bolsigdb):
    """Energy (eV) given to the electron in each process of bolsigdb.dat, by
    (target, product): -threshold for excitation and ionization, +threshold for
    the reverse direction of a process X <-> Y, zero for attachment (the LXCat
    format gives no threshold)."""
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
                    raise ErroQuimica(f"{bolsigdb}: process {proc} without the threshold line (eV)") from None
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


# Fortran functions accepted in rate expressions, and the variables that
# ZDPlasKin_reac_rates receives: TGAS and TE in K, EN in Td
FUNCOES = {"exp": np.exp, "sqrt": np.sqrt, "log": np.log, "log10": np.log10, "abs": np.abs,
           "max": _maximo, "min": _minimo, "dexp": np.exp, "dsqrt": np.sqrt, "dlog": np.log}
ENTRADAS = ("tgas", "te", "en", "any_neutral")


# numeric literal of Fortran: integer, single precision real (with a point or an
# E exponent) or double precision real (D exponent)
_LITERAL = re.compile(r"(?<![\w.])(\d+\.\d*|\.\d+|\d+)(?:([eEdD])([+-]?\d+))?(?![\w.])")


def _literal_python(m):
    """Fortran literal in Python with the same value: a real without a D exponent
    is single precision in Fortran (0.39 is 0.38999998569...), so it enters
    rounded to float32; one with a D exponent is double; an integer stays integer."""
    mantissa, letra, expoente = m.group(1), m.group(2), m.group(3)
    if letra is None and "." not in mantissa:
        return mantissa
    texto = mantissa + ("e" + expoente if letra else "")
    if letra in ("d", "D"):
        return repr(float(texto))
    # marked as single precision: operations among such numbers are single precision too
    return f"_s({float(np.float32(float(texto)))!r})"


def _expressao_python(expr):
    """Fortran expression of ZDPlasKin_reac_rates in Python with numpy: literals
    with the value Fortran uses (single, double, integer) and names in lower case
    (Fortran does not distinguish)."""
    e = _LITERAL.sub(_literal_python, expr)
    return re.sub(r"(?<!\d)[A-Za-z_][A-Za-z_0-9]*", lambda m: m.group(0).lower(), e)



# relational and logical operators of Fortran, already in lower case
_OPERADORES = ((".gt.", " > "), (".lt.", " < "), (".ge.", " >= "), (".le.", " <= "), (".eq.", " == "),
               (".ne.", " != "), ("/=", " != "), (".and.", " and "), (".or.", " or "), (".not.", " not "),
               (".true.", " True "), (".false.", " False "))

# ZDPlasKin_cfg entries meaningful in the rate routine: T_g, E/N and T_e
_CFG = {1: "tgas", 3: "en", 4: "te"}


def _divisao_fortran(a, b):
    """Integer division of Fortran: truncated towards zero."""
    q = abs(a) // abs(b)
    return q if (a >= 0) == (b >= 0) else -q


class _Fortran(ast.NodeTransformer):
    """Fortran semantics on the expression tree: element-wise and, or and not
    (numpy) and the types of the numbers. Operations among integers are integer
    (division truncated: 1/2 is 0); operations among single precision reals, or
    between single and integer, are single and rounded to float32 (1.0/3.0 is
    0.33333334); with a double operand (a literal with D or any variable, all
    double precision in the rate routine) the operation is double. The
    precedence of Python in *, / and ** is that of Fortran (x*3/2 is (x*3)/2, real)."""

    def __init__(self):
        self.divisoes_inteiras = 0

    @staticmethod
    def _tipo(no):
        """'i' integer, 's' single precision real or 'd' double."""
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
    """Index of the parenthesis that closes the condition of an IF; a clear error if it does not close."""
    try:
        return _fecha_parenteses(linha, inicio)
    except ValueError:
        raise ErroQuimica(f"IF with an unclosed parenthesis: {linha}") from None


def _fecha_parenteses(texto, inicio):
    """Index of the parenthesis that closes the one opened at texto[inicio]."""
    nivel = 0
    for i in range(inicio, len(texto)):
        nivel += (texto[i] == "(") - (texto[i] == ")")
        if nivel == 0:
            return i
    raise ValueError("unclosed parenthesis")


def _linhas_logicas(bloco):
    """Lines of the block with the continuations (& at the end) joined."""
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
    """Parts of the text separated by the commas that are not inside parentheses."""
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
        # coefficients: column of the table (k of BOLSIG+) or expression
        taxas = _bloco(texto, "subroutine ZDPlasKin_reac_rates", "end subroutine ZDPlasKin_reac_rates")
        self.coluna_bolsig = [None] * nr
        self.avisos = []
        self._le_programa(_linhas_logicas(taxas)[1:])
        # reactant orders: rrt(i) = rrt(i) * density(a)**p * density(b) ...
        fex = _bloco(texto, "subroutine ZDPlasKin_fex", "end subroutine ZDPlasKin_fex")
        self.ordem = np.zeros((ne, nr))
        for m in re.finditer(r"rrt\((\d+)\)\s*=\s*rrt\(\d+\)\s*\*(.+)", fex):
            i = int(m.group(1)) - 1
            for d in re.finditer(r"density\((\d+)\)(\*\*(\d+))?", m.group(2)):
                self.ordem[int(d.group(1)) - 1, i] += int(d.group(3) or 1)
        # stoichiometry: reac_source_local(s,r) = +- reac_rate_local(r) [* f]
        matriz = _bloco(texto, "subroutine ZDPlasKin_reac_source_matrix",
                        "end subroutine ZDPlasKin_reac_source_matrix")
        self.estequiometria = np.zeros((ne, nr))
        for m in re.finditer(r"reac_source_local\((\d+),(\d+)\)\s*=\s*([+-])\s*reac_rate_local\(\d+\)"
                             r"(\s*\*\s*([\d.]+)[dD]0)?", matriz):
            f = float(m.group(5)) if m.group(5) else 1.0
            self.estequiometria[int(m.group(1)) - 1, int(m.group(2)) - 1] = (-f if m.group(3) == "-" else f)
        self.indice = {nome.upper(): k for k, nome in enumerate(self.especies)}
        # BOLSIG+ reactions with two or more reacting electrons: the BOLSIG+ rate
        # is a two-body one (cm3/s) and cannot be a three-body coefficient
        ie = self.indice["E"]
        self.bolsig_tres_corpos = [self.reacoes[i] for i in range(nr)
                                   if self.coluna_bolsig[i] is not None and self.ordem[ie, i] >= 2]

    def energia_eletronica(self, bolsigdb, fundo):
        """Energy (eV) that each electron impact reaction gives to the electron,
        for reactions whose target is not the background gas, whose losses the
        table of the background does not contain: -threshold for excitation and
        ionization, +threshold for superelastic collisions (a process X <-> Y of
        bolsigdb.dat taken from Y to X). Zero for the other reactions (those of the
        background are already in the tabulated power). fundo: name of the
        background species, or a set of names in a mixture. bolsigdb None: the
        thresholds stored in quimica.json."""
        fundos = {fundo.upper()} if isinstance(fundo, str) else {f.upper() for f in fundo}
        processos = (getattr(self, "_limiares", None) or {}) if bolsigdb is None else _le_limiares(bolsigdb)
        energia = np.zeros(len(self.reacoes))
        for i, r in enumerate(self.reacoes):
            if self.coluna_bolsig[i] is None:
                continue
            a, b = r.split(":", 1)[1].upper().split("->")
            if a not in fundos:
                if (a, b) not in processos:
                    onde = "in quimica.json" if bolsigdb is None else f"in {bolsigdb}"
                    raise ErroQuimica(f"reaction {i + 1} ({r}): no threshold for {a} -> {b} {onde}")
                energia[i] = processos[(a, b)]
        return energia

    def salva(self, caminho, bolsigdb=None):
        """Writes the chemistry as JSON (quimica.json): species, charges,
        reactions, orders, stoichiometry and the rate program as Python text, plus
        the thresholds of bolsigdb.dat for the BOLSIG+ reactions. The file holds
        no ZDPlasKin code: it is kinet.inp translated. The keys keep their
        original (Portuguese) names, which are part of the file format."""
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
        """Chemistry written by salva, without zdplaskin_m.F90."""
        d = json.loads(pathlib.Path(caminho).read_text())
        if d.get("formato") != "quimica.json 1":
            raise ErroQuimica(f"{caminho}: unknown format {d.get('formato')!r} (expected 'quimica.json 1')")
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
        """Program of the rate routine: (target, code) in order, the target being the
        name of a local variable or the index of a reaction; inside IF the code is
        np.where(condition, expression, previous value)."""
        self._programa = []
        self._textos = []                # Python text of each step (None: BOLSIG+ column), for the export
        conhecidos = set(FUNCOES) | set(ENTRADAS)
        definidas = set()
        rrt_definidas = set()
        pilha = []                       # per open IF: [condition of the branch, any previous branch]
        expressoes = set()               # reactions with a rate given by an expression
        # variables the reader cannot compute (ZDPlasKin_cfg other than 1, 3 and
        # 4, such as the De the preprocessor emits): an error only if used
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
                return          # ANY_NEUTRAL: input of coefficients, the density of neutrals
            if minus.startswith("call ") and not minus.startswith("call zdplaskin_bolsig_rates"):
                raise ErroQuimica(f"line of ZDPlasKin_reac_rates with a call, which the reader does not execute: {linha}")
            if "::" in linha:
                # declaration; with a value (parameter) it becomes a local variable;
                # several declarations separated by commas outside parentheses
                for item in _separa_no_nivel_zero(linha.split("::", 1)[1]):
                    if "=" in item:
                        nome, valor = (x.strip() for x in item.split("=", 1))
                        atribui(nome.lower(), valor, f"variable {nome}")
                return
            if minus == "return" and pilha:
                raise ErroQuimica(f"RETURN inside IF in ZDPlasKin_reac_rates, which the reader does not follow: {linha}")
            if (not linha or minus.startswith(("use ", "implicit", "call ", "where", "return", "lreaction_block"))
                    or padrao_ent.match(linha)):
                return
            m = re.match(r"(?i)if\s*\(", linha)
            if m:
                fim = _fim_da_condicao(linha, m.end() - 1)
                cond = self._traduz(linha[m.end():fim], f"condition in {linha}", conhecidos)
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
                    raise ErroQuimica(f"ELSE IF without IF: {linha}")
                fim = _fim_da_condicao(linha, m.end() - 1)
                cond = self._traduz(linha[m.end():fim], f"condition in {linha}", conhecidos)
                anterior = pilha[-1][1]
                pilha[-1] = [f"_e(_nao({anterior}), {cond})", f"_ou({anterior}, {cond})"]
                return
            if minus == "else":
                if not pilha:
                    raise ErroQuimica("ELSE without IF")
                pilha[-1][0] = f"_nao({pilha[-1][1]})"
                return
            if minus.replace(" ", "") == "endif":
                if not pilha:
                    raise ErroQuimica("END IF without IF")
                pilha.pop()
                return
            m = re.match(r"(?i)rrt\((\d+)\)\s*=\s*(.+)", linha)
            if m:
                i = int(m.group(1)) - 1
                direita = m.group(2).strip()
                b = re.match(r"bolsig_rates\(bolsig_pointer\((\d+)\)\)$", direita)
                if b:
                    if pilha:
                        raise ErroQuimica(f"reaction {i + 1}: BOLSIG+ rate inside IF, which the reader does not "
                                          f"separate per cell in the energy and heat terms: {linha}")
                    self.coluna_bolsig[i] = int(b.group(1)) - 1
                    self._programa.append((i, int(b.group(1)) - 1, f"reaction {i + 1}"))
                    self._textos.append(None)
                    rrt_definidas.add(i)
                    return
                # rrt(j): the coefficient of an earlier reaction (before the densities)
                for j in re.findall(r"(?i)rrt\((\d+)\)", direita):
                    if int(j) - 1 >= i:
                        raise ErroQuimica(f"reaction {i + 1}: uses the rate of reaction {int(j)}, which comes later: {linha}")
                direita = re.sub(r"(?i)rrt\((\d+)\)", lambda r: f"rrt_k[{int(r.group(1)) - 1}]", direita)
                atribui(i, direita, f"reaction {i + 1}")
                return
            m = re.match(r"([A-Za-z_][A-Za-z_0-9]*)\s*=\s*(.+)$", linha)
            if not m:
                raise ErroQuimica(f"line of ZDPlasKin_reac_rates that the reader cannot evaluate: {linha}")
            atribui(m.group(1).lower(), m.group(2), f"variable {m.group(1)}")

        for linha in linhas:
            instrucao(linha)
        if pilha:
            raise ErroQuimica("IF without END IF in ZDPlasKin_reac_rates")
        mistas = sorted(i + 1 for i in expressoes if self.coluna_bolsig[i] is not None)
        if mistas:
            raise ErroQuimica(f"reactions {', '.join(map(str, mistas))}: BOLSIG+ rate redefined by an expression; "
                              "the reader treats the whole reaction as a BOLSIG+ one in the energy and heat terms")
        codigos = [c for _, c, _ in self._programa if not isinstance(c, int)]
        self.usa_en = any("en" in c.co_names for c in codigos)
        self.usa_any_neutral = any("any_neutral" in c.co_names for c in codigos)

    def _traduz(self, texto, onde, conhecidos):
        """Fortran expression (or condition) as valid Python code, with the names
        checked; integer division becomes that of Fortran, with a warning."""
        texto = texto.strip()
        e = _expressao_python(texto)
        for fortran, python in _OPERADORES:
            e = e.replace(fortran, python)
        e = re.sub(r"zdplaskin_cfg\((\d+)\)", lambda m: _CFG.get(int(m.group(1)), m.group(0).upper()), e).strip()
        if "ZDPLASKIN_CFG" in e:
            usados = ", ".join(sorted(set(re.findall(r"ZDPLASKIN_CFG\(\d+\)", e))))
            raise ErroQuimica(f"{onde}: {usados} in "
                              f"{texto}; in the rate routine the reader knows ZDPlasKin_cfg(1), (3) and (4), "
                              "T_g, E/N and T_e")
        fortran = _Fortran()
        try:
            arvore = fortran.visit(ast.parse(e, mode="eval"))
        except SyntaxError as erro:
            raise ErroQuimica(f"{onde}: expression that the reader cannot evaluate: {texto} ({erro.msg})") from None
        if fortran.divisoes_inteiras:
            self.avisos.append(f"{onde}: integer division in {texto}, done as in Fortran "
                               "(truncated: 1/2 is 0); write 1.0/2.0 if a real division was meant")
        codigo = compile(ast.fix_missing_locations(arvore), onde, "eval")
        desconhecidos = sorted(set(codigo.co_names) - conhecidos - set(AUXILIARES) - {"_divisao_fortran"})
        for nome in desconhecidos:
            if nome in getattr(self, "_indisponiveis", {}):
                raise ErroQuimica(f"{onde} uses {nome.upper()}, which the reader does not compute: "
                                  f"{self._indisponiveis[nome]}")
        if desconhecidos:
            raise ErroQuimica(f"{onde}: unknown name {', '.join(n.upper() for n in desconhecidos)} in {texto}")
        return ast.unparse(arvore)

    def coeficientes(self, k_bolsig, tgas, te, en=None, any_neutral=None):
        """Rate coefficients per cell. k_bolsig: (cells, columns) of the table;
        tgas and te in K, en (local E/N, Td) and any_neutral (total density of
        neutrals, cm-3, the ALL_NEUTRAL of ZDPlasKin) per cell. Returns
        (reactions, cells)."""
        if en is None and self.usa_en:
            raise ErroQuimica("the chemistry uses EN (local E/N): pass en to coeficientes")
        if any_neutral is None and self.usa_any_neutral:
            raise ErroQuimica("the chemistry uses ANY_NEUTRAL (density of neutrals): pass any_neutral to coeficientes")
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
            # with finite inputs, NaN is an undefined value: a variable or rate
            # assigned in one IF branch only (a bad step of the integrator, with
            # NaN input, passes on for the integrator to reject)
            r = int(np.nonzero(np.isnan(k).any(axis=1))[0][0])
            raise ErroQuimica(f"reaction {r + 1}: coefficient without value in part of the cells (variable or rate "
                              "assigned in one IF branch only, without ELSE or an earlier value)")
        return k

    def taxas(self, dens, k):
        """Rate of each reaction (reactions, cells); dens in cm-3 (species, cells)."""
        produto = np.ones_like(k)
        for s in range(len(self.especies)):
            for i in np.nonzero(self.ordem[s])[0]:
                produto[i] *= dens[s] ** self.ordem[s, i]
        return k * produto

    def fontes(self, dens, k):
        """Source terms (species, cells) in cm-3/s."""
        return self.estequiometria @ self.taxas(dens, k)
