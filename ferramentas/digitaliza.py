# SPDX-License-Identifier: GPL-3.0-or-later
"""Digitalização de figuras de artigos para comparação com o simulador.

Duas fontes. Figura vetorial: a página do PDF vira SVG pelo pdftocairo e cada
traço sai com cor e coordenadas exatas (pt, y para baixo). Figura raster: o PNG
(da página renderizada ou da imagem embutida) é lido por pixel, e as séries são
separadas pela cor.

Em ambos os casos a moldura dos eixos é achada pelas linhas escuras longas,
como retângulo (x0, y_topo, x1, y_base), e a calibração é feita por Eixo, que
liga duas coordenadas a dois valores conhecidos (os limites da moldura ou dois
traços de escala lidos na figura). Os scripts que usam este módulo ficam em
docs/literatura/dados/ e registram cada calibração."""
import re
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

_NS = "{http://www.w3.org/2000/svg}"


class Eixo:
    """Coordenada na figura para valor: linear ou logarítmico (base 10), a
    partir de duas referências (c1, v1) e (c2, v2)."""

    def __init__(self, c1, v1, c2, v2, log=False):
        self.log = log
        f = np.log10 if log else (lambda v: v)
        self.c1, self.c2, self.f1, self.f2 = float(c1), float(c2), f(float(v1)), f(float(v2))

    def valor(self, c):
        f = self.f1 + (np.asarray(c, dtype=float) - self.c1) * (self.f2 - self.f1) / (self.c2 - self.c1)
        return 10.0 ** f if self.log else f


# ---------------------------------------------------------------- vetorial

def _cor(texto):
    m = re.match(r"rgb\(\s*([\d.]+)%\s*,\s*([\d.]+)%\s*,\s*([\d.]+)%\s*\)", texto or "")
    return tuple(float(g) / 100.0 for g in m.groups()) if m else None


def _matriz(t):
    m = np.eye(3)
    for nome, args in re.findall(r"(\w+)\(([^)]*)\)", t or ""):
        a = [float(x) for x in re.split(r"[ ,]+", args.strip()) if x]
        if nome == "matrix":
            k = np.array([[a[0], a[2], a[4]], [a[1], a[3], a[5]], [0.0, 0.0, 1.0]])
        elif nome == "translate":
            k = np.array([[1.0, 0.0, a[0]], [0.0, 1.0, a[1] if len(a) > 1 else 0.0], [0.0, 0.0, 1.0]])
        elif nome == "scale":
            k = np.diag([a[0], a[1] if len(a) > 1 else a[0], 1.0])
        else:
            continue
        m = m @ k
    return m


def _subcaminhos(d):
    """Pontos de cada subcaminho (M ... L/C ... Z); das curvas de Bézier só o
    ponto final, o que basta para marcadores e polilinhas."""
    toks = re.findall(r"[MLCZHVmlczhv]|-?[\d.]+(?:e-?\d+)?", d)
    out, sub, i, cmd = [], [], 0, None
    while i < len(toks):
        t = toks[i]
        if t.isalpha():
            cmd = t
            i += 1
            if cmd in "Zz" and sub:
                # Z fecha o subcaminho: o último lado volta ao primeiro ponto
                if len(sub) > 2 and sub[-1] != sub[0]:
                    sub.append(sub[0])
                out.append(sub)
                sub = []
            continue
        if cmd == "M":
            if sub:
                out.append(sub)
            sub = [(float(toks[i]), float(toks[i + 1]))]
            i += 2
            cmd = "L"
        elif cmd == "L":
            sub.append((float(toks[i]), float(toks[i + 1])))
            i += 2
        elif cmd == "C":
            sub.append((float(toks[i + 4]), float(toks[i + 5])))
            i += 6
        else:
            i += 1
    if sub:
        out.append(sub)
    return out


def _propriedade(el, nome, herdada):
    """Valor de stroke ou fill: o pdftocairo até o 22 grava no atributo style,
    o 24 (cairo 1.18) em atributo próprio; sem nenhum dos dois, vale o do
    elemento de cima, como no SVG."""
    v = re.search(nome + r":\s*(rgb\([^)]*\)|none)", el.get("style", ""))
    if v:
        return v.group(1)
    return el.get(nome, herdada)


def _percorre(el, m, saida, traco=None, preench=None):
    m = m @ _matriz(el.get("transform"))
    traco = _propriedade(el, "stroke", traco)
    preench = _propriedade(el, "fill", preench)
    if el.tag == _NS + "path":
        for sub in _subcaminhos(el.get("d", "")):
            p = np.array([(m @ [x, y, 1.0])[:2] for x, y in sub])
            saida.append({"traco": _cor(traco), "preenchimento": _cor(preench), "p": p})
    for f in el:
        _percorre(f, m, saida, traco, preench)


def tracos_pdf(pdf, pagina):
    """Traços da página (1 = primeira) como dicionários com 'traco' e
    'preenchimento' (cor RGB de 0 a 1, ou None) e 'p' (pontos, pt). Os glifos
    do texto ficam em <defs> e são ignorados."""
    with tempfile.TemporaryDirectory() as d:
        svg = Path(d) / "p.svg"
        subprocess.run(["pdftocairo", "-svg", "-f", str(pagina), "-l", str(pagina), str(pdf), str(svg)],
                       check=True, capture_output=True)
        return tracos_svg(svg)


def tracos_svg(svg):
    """Traços de um SVG gravado pelo pdftocairo, no formato de tracos_pdf."""
    raiz = ET.parse(svg).getroot()
    saida = []
    for f in raiz:
        if f.tag != _NS + "defs":
            _percorre(f, np.eye(3), saida)
    return saida


def _mesma_cor(a, b, tol=0.08):
    return a is not None and max(abs(x - y) for x, y in zip(a, b)) <= tol


def _dentro(p, janela, folga=0.0):
    x0, y0, x1, y1 = janela
    return (p[:, 0] >= x0 - folga) & (p[:, 0] <= x1 + folga) & (p[:, 1] >= y0 - folga) & (p[:, 1] <= y1 + folga)


def _retangulos(horiz, vert, tol, min_lado=10.0):
    """Retângulos formados por segmentos horizontais (y, x0, x1) e verticais
    (x, y0, y1): dois verticais e dois horizontais em que cada um cobre o
    intervalo entre os outros dois. Cobrir, e não coincidir, porque traços de
    escala nos limites dos eixos emendam nas bordas da moldura."""
    xs = sorted({round(v[0], 1) for v in vert})
    ys = sorted({round(h[0], 1) for h in horiz})
    rets = []
    for i, xa in enumerate(xs):
        for xb in xs[i + 1:]:
            if xb - xa < min_lado:
                continue
            for j, ya in enumerate(ys):
                for yb in ys[j + 1:]:
                    if yb - ya < min_lado:
                        continue
                    ok = all(any(abs(h[0] - y) <= tol and h[1] <= xa + tol and h[2] >= xb - tol for h in horiz)
                             for y in (ya, yb))
                    ok = ok and all(any(abs(v[0] - x) <= tol and v[1] <= ya + tol and v[2] >= yb - tol
                                        for v in vert) for x in (xa, xb))
                    if ok and not any(np.allclose((xa, ya, xb, yb), r, atol=tol) for r in rets):
                        # a mesma moldura desenhada duas vezes, com décimos de ponto de diferença, conta uma
                        rets.append((xa, ya, xb, yb))
    # uma moldura com linhas de grade dentro gera retângulos internos; fica o externo
    return [r for r in rets if not any(s != r and s[0] <= r[0] + tol and s[2] >= r[2] - tol and
                                       s[1] <= r[1] + tol and s[3] >= r[3] - tol for s in rets)]


def molduras_vetoriais(tracos, min_lado=30.0, tol=0.6):
    """Molduras dos eixos (x0, y_topo, x1, y_base) em pt."""
    horiz, vert = [], []
    for t in tracos:
        if not _mesma_cor(t["traco"], (0.0, 0.0, 0.0), 0.3):
            continue
        p = t["p"]
        for (xa, ya), (xb, yb) in zip(p[:-1], p[1:]):
            if abs(ya - yb) < 0.05 and abs(xa - xb) >= min_lado:
                horiz.append((ya, min(xa, xb), max(xa, xb)))
            elif abs(xa - xb) < 0.05 and abs(ya - yb) >= min_lado:
                vert.append((xa, min(ya, yb), max(ya, yb)))
    return _retangulos(horiz, vert, tol)


def tracos_de_escala(tracos, moldura, eixo, min_comp=3.0, max_comp=12.0, tol=0.6, cor=(0.0, 0.0, 0.0)):
    """Posições (pt, em ordem crescente) dos traços de escala do eixo 'x'
    (borda de baixo da moldura) ou 'y' (borda da esquerda): segmentos
    perpendiculares à borda, encostados nela, com comprimento entre min_comp e
    max_comp. Com min_comp maior que os traços menores, só os principais."""
    x0, y0, x1, y1 = moldura
    pos = []
    for t in tracos:
        if not _mesma_cor(t["traco"], cor, 0.3):
            continue
        p = t["p"]
        for (xa, ya), (xb, yb) in zip(p[:-1], p[1:]):
            if eixo == "x" and abs(xa - xb) < 0.05 and min_comp <= abs(ya - yb) <= max_comp \
                    and min(abs(ya - y1), abs(yb - y1)) <= tol and x0 - tol <= xa <= x1 + tol:
                pos.append(xa)
            elif eixo == "y" and abs(ya - yb) < 0.05 and min_comp <= abs(xa - xb) <= max_comp \
                    and min(abs(xa - x0), abs(xb - x0)) <= tol and y0 - tol <= ya <= y1 + tol:
                pos.append(ya)
    unicos = []
    for v in sorted(pos):
        if not unicos or v - unicos[-1] > 0.3:
            unicos.append(v)
    return np.array(unicos)


def marcadores_vetoriais(tracos, cor, janela, max_tam=12.0, min_tam=0.8):
    """Centros (pt) dos marcadores da cor dada dentro da janela: subcaminhos
    pequenos, pelo centro da caixa envolvente; centros a menos de 1 pt são o
    mesmo marcador (contorno e preenchimento desenhados à parte)."""
    centros = []
    for t in tracos:
        if not (_mesma_cor(t["traco"], cor) or _mesma_cor(t["preenchimento"], cor)):
            continue
        p = t["p"]
        tam = p.max(0) - p.min(0)
        if len(p) < 3 or tam.max() > max_tam or tam.max() < min_tam:
            continue
        c = 0.5 * (p.max(0) + p.min(0))
        if _dentro(c[None, :], janela, 1.0)[0] and not any(np.hypot(*(c - q)) < 1.0 for q in centros):
            centros.append(c)
    return np.array(centros).reshape(-1, 2)


def curva_vetorial(tracos, cor, janela, min_pontos=10):
    """Pontos (pt) das polilinhas da cor dada com pelo menos min_pontos
    vértices, cortados à janela e ordenados em x."""
    partes = [t["p"] for t in tracos if _mesma_cor(t["traco"], cor) and len(t["p"]) >= min_pontos]
    if not partes:
        return np.zeros((0, 2))
    p = np.concatenate(partes)
    p = p[_dentro(p, janela)]
    return p[np.argsort(p[:, 0], kind="stable")]


# ---------------------------------------------------------------- raster

def le_png(caminho):
    """Imagem como matriz (altura, largura, 3) de 0 a 1."""
    from PIL import Image
    return np.asarray(Image.open(caminho).convert("RGB"), dtype=float) / 255.0


def pagina_png(pdf, pagina, destino, dpi=300):
    """Renderiza uma página do PDF em PNG e devolve o caminho."""
    destino = Path(destino)
    subprocess.run(["pdftoppm", "-r", str(dpi), "-f", str(pagina), "-l", str(pagina), "-singlefile", "-png",
                    str(pdf), str(destino.with_suffix(""))], check=True, capture_output=True)
    return destino.with_suffix(".png")


def _corridas(linha):
    """Intervalos [i0, i1] de valores verdadeiros consecutivos."""
    d = np.diff(np.concatenate([[0], linha.astype(int), [0]]))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1) - 1))


def _segmentos(escuro, min_lado):
    """Segmentos longos de pixels escuros por linha, agrupando linhas vizinhas
    de uma mesma reta grossa no centro dela: (y, x0, x1)."""
    abertos, fechados = [], []
    for y in range(escuro.shape[0]):
        continuam = []
        for i0, i1 in _corridas(escuro[y]):
            if i1 - i0 + 1 < min_lado:
                continue
            g = next((g for g in abertos if abs(g[-1][1] - i0) <= 2 and abs(g[-1][2] - i1) <= 2), None)
            if g is None:
                g = []
            else:
                abertos.remove(g)
            g.append((y, i0, i1))
            continuam.append(g)
        fechados += abertos
        abertos = continuam
    return [(float(np.mean([s[0] for s in g])), float(np.median([s[1] for s in g])),
             float(np.median([s[2] for s in g]))) for g in fechados + abertos]


def molduras_raster(img, min_frac=0.12, limiar=0.35, tol=3.0):
    """Molduras dos eixos (x0, y_topo, x1, y_base) em pixels."""
    escuro = img.sum(axis=2) / 3.0 < limiar
    horiz = _segmentos(escuro, int(min_frac * img.shape[1]))
    vert = [(x, y0, y1) for x, y0, y1 in _segmentos(escuro.T, int(min_frac * img.shape[0]))]
    return _retangulos(horiz, vert, tol)


def mascara(img, cor, tol=0.25, janela=None, folga=3):
    """Pixels de cor próxima da dada (distância máxima por canal), opcionalmente
    só dentro da janela encolhida de folga pixels (para não pegar a moldura)."""
    m = np.max(np.abs(img - np.asarray(cor)[None, None, :]), axis=2) <= tol
    if janela is not None:
        x0, y0, x1, y1 = janela
        dentro = np.zeros_like(m)
        dentro[int(np.ceil(y0)) + folga:int(np.floor(y1)) - folga + 1,
               int(np.ceil(x0)) + folga:int(np.floor(x1)) - folga + 1] = True
        m &= dentro
    return m


def marcadores_raster(img, cor, janela, tol=0.25, min_px=6, abertura=0, separa=False):
    """Centroides (pixels) das manchas da cor dada dentro da janela. Com
    abertura k > 0, uma abertura morfológica k × k apaga antes as linhas mais
    finas que k (barras de erro, curvas de modelo na mesma cor). Com separa,
    uma mancha com área de n marcadores típicos (a mediana das áreas) vira n
    marcadores, pelo k-médias dos seus pixels: marcadores que se tocam."""
    from scipy import ndimage
    m = mascara(img, cor, tol, janela)
    if abertura:
        m = ndimage.binary_opening(m, np.ones((abertura, abertura), dtype=bool))
    rot, n = ndimage.label(m)
    if n == 0:
        return np.zeros((0, 2))
    tam = ndimage.sum(m, rot, range(1, n + 1))
    cy_cx = ndimage.center_of_mass(m, rot, range(1, n + 1))
    tipico = float(np.median(tam[tam >= min_px])) if np.any(tam >= min_px) else 1.0
    caixas = ndimage.find_objects(rot)
    extensao = np.array([max(c[0].stop - c[0].start, c[1].stop - c[1].start) for c in caixas], dtype=float)
    diametro = float(np.median(extensao[tam >= min_px])) if np.any(tam >= min_px) else 1.0
    centros = []
    for k, ((cy, cx), t) in enumerate(zip(cy_cx, tam), start=1):
        if t < min_px:
            continue
        # n marcadores pela área e pelo comprimento: um resto de barra de erro
        # aumenta a área sem alongar a mancha como dois marcadores lado a lado
        quantos = min(int(round(t / tipico)), int(round(extensao[k - 1] / diametro))) if separa else 1
        if quantos <= 1:
            centros.append((cx, cy))
            continue
        from scipy.cluster.vq import kmeans2
        ys, xs = np.nonzero(rot == k)
        pts = np.column_stack([xs, ys]).astype(float)
        # sementes espalhadas ao longo da maior extensão da mancha
        eixo = 0 if np.ptp(xs) >= np.ptp(ys) else 1
        ordem = np.argsort(pts[:, eixo])
        sementes = pts[ordem[np.linspace(0, len(pts) - 1, quantos).astype(int)]]
        cc, _ = kmeans2(pts, sementes, minit="matrix", iter=50)
        centros.extend(map(tuple, cc))
    return np.array(centros).reshape(-1, 2)


def curva_raster(img, cor, janela, tol=0.25, maior_componente=False):
    """Curva da cor dada: para cada coluna com pixels da cor, a mediana das
    linhas (x, y em pixels). Com maior_componente, só os componentes conexos
    grandes (vizinhança de 8, ao menos 10 % do maior): separa uma curva contínua,
    mesmo partida onde outra curva a cruza, de um tracejado da mesma cor, que se
    parte em pedaços pequenos."""
    m = mascara(img, cor, tol, janela)
    if maior_componente:
        from scipy import ndimage
        # rotulada sobre a máscara dilatada de um pixel, que fecha as falhas da
        # compressão em trechos íngremes sem emendar os traços do tracejado
        cheia = ndimage.binary_dilation(m, structure=np.ones((3, 3), dtype=bool))
        rot, n = ndimage.label(cheia, structure=np.ones((3, 3), dtype=bool))
        if n:
            # a curva contínua se parte onde outra curva a cruza: ficam todos os
            # pedaços com pelo menos 10 % do maior; os traços do tracejado são menores
            tam = ndimage.sum(cheia, rot, range(1, n + 1))
            grandes = 1 + np.flatnonzero(tam >= 0.1 * tam.max())
            m &= np.isin(rot, grandes)
    xs = np.flatnonzero(m.any(axis=0))
    return np.array([(x, np.median(np.flatnonzero(m[:, x]))) for x in xs]).reshape(-1, 2)


def espessura_raster(img, cor, janela, tol=0.25):
    """Espessura vertical típica da linha (pixels): mediana do comprimento das
    corridas de pixels da cor nas colunas com uma corrida só."""
    m = mascara(img, cor, tol, janela)
    comp = [c[0][1] - c[0][0] + 1 for c in (_corridas(m[:, x]) for x in np.flatnonzero(m.any(axis=0)))
            if len(c) == 1]
    return float(np.median(comp)) if comp else 1.0


def extremo_raster(img, cor, janela, sentido="max", tol=0.25):
    """Pico (sentido 'max', o mais alto na figura) ou vale ('min') da curva:
    a borda externa da linha deslocada de meia espessura para dentro, (x, y)
    em pixels. A mediana por coluna (curva_raster) subestima picos estreitos,
    em que a coluna mistura a subida e a descida."""
    m = mascara(img, cor, tol, janela)
    ys, xs = np.nonzero(m)
    meia = 0.5 * espessura_raster(img, cor, janela, tol)
    k = np.argmin(ys) if sentido == "max" else np.argmax(ys)
    alvo = ys[k]
    x = xs[ys == alvo].mean()
    return float(x), float(alvo + meia if sentido == "max" else alvo - meia)


def tracos_de_escala_raster(img, pos, eixo, intervalo, lado="fora", limiar=0.45, min_comp=3):
    """Posições (pixels) dos traços de escala ao longo de um eixo raster: o eixo
    'x' é a linha horizontal na linha de pixels pos, o 'y' a vertical na coluna
    pos; intervalo (c0, c1) limita a busca ao longo do eixo. lado 'fora' procura
    traços abaixo do eixo x ou à esquerda do y, 'dentro' do outro lado. Um traço
    é uma coluna (ou linha) escura por min_comp pixels logo depois da espessura
    do eixo."""
    a = img.mean(axis=2) < limiar
    if eixo == "y":
        a = a.T
    c0, c1 = int(np.floor(intervalo[0])), int(np.ceil(intervalo[1])) + 1
    r = int(round(pos))
    faixa = lambda k: 0 <= k < a.shape[0] and a[k, c0:c1].mean() > 0.5
    r0, r1 = r, r
    while faixa(r0 - 1):
        r0 -= 1
    while faixa(r1 + 1):
        r1 += 1
    sentido = (1 if eixo == "x" else -1) * (1 if lado == "fora" else -1)
    inicio = r1 + 1 if sentido > 0 else r0 - 1
    linhas = [inicio + sentido * k for k in range(min_comp)]
    if min(linhas) < 0 or max(linhas) >= a.shape[0]:
        return np.zeros(0)
    escuras = np.all(a[linhas, :], axis=0)
    escuras[:c0] = False
    escuras[c1:] = False
    return np.array([0.5 * (i0 + i1) for i0, i1 in _corridas(escuras)])


def mascara_alfa(img, cor, alfa_min=0.2, tol=0.15, janela=None):
    """Pixels que são a cor dada misturada com o branco do fundo, com opacidade
    de pelo menos alfa_min: p = 1 - a (1 - cor). Pega traços claros, como
    barras de erro semitransparentes, que a distância à cor cheia não pega."""
    d = 1.0 - img
    u = 1.0 - np.asarray(cor, dtype=float)
    a = (d @ u) / (u @ u)
    resto = np.linalg.norm(d - a[..., None] * u[None, None, :], axis=2)
    m = (a >= alfa_min) & (resto <= tol)
    if janela is not None:
        x0, y0, x1, y1 = janela
        dentro = np.zeros_like(m)
        dentro[int(np.ceil(y0)):int(np.floor(y1)) + 1, int(np.ceil(x0)):int(np.floor(x1)) + 1] = True
        m &= dentro
    return m


def barras_raster(img, cor, centros, janela, alcance, tol=0.25, meia_largura=1, alfa_min=None):
    """Extremos (pixels) das barras de erro verticais em cada centro: a partir
    do centro do marcador, a corrida contínua de pixels da cor na coluna (e nas
    meia_largura vizinhas) para cima e para baixo, até alcance pixels. Com
    alfa_min, a cor vale também misturada ao branco (barras claras, ver
    mascara_alfa). Sem barra, os extremos são as bordas do próprio marcador."""
    m = mascara(img, cor, tol, janela, folga=0)
    if alfa_min is not None:
        m |= mascara_alfa(img, cor, alfa_min, janela=janela)
    topo, base = [], []
    for cx, cy in centros:
        x0, x1 = int(round(cx)) - meia_largura, int(round(cx)) + meia_largura + 1
        col = m[:, x0:x1].any(axis=1)
        y = int(round(cy))
        t = y
        while t - 1 >= max(0, y - alcance) and col[t - 1]:
            t -= 1
        b = y
        while b + 1 <= min(len(col) - 1, y + alcance) and col[b + 1]:
            b += 1
        topo.append(t)
        base.append(b)
    return np.array(topo, dtype=float), np.array(base, dtype=float)


def curva_raster_seguida(img, cor, janela, tol=0.25, salto=25.0):
    """Curva da cor dada seguida de coluna em coluna, para separar duas curvas da
    mesma cor (medida contínua e ajuste tracejado, por exemplo): em cada coluna,
    as corridas de pixels da cor; fica a corrida cujo centro está mais perto da
    previsão linear pelas duas colunas anteriores, se a menos de salto pixels.
    Começa na primeira coluna com uma corrida só. (x, y) em pixels."""
    m = mascara(img, cor, tol, janela)
    pts = []
    for x in np.flatnonzero(m.any(axis=0)):
        centros = [0.5 * (a + b) for a, b in _corridas(m[:, x])]
        if not pts:
            if len(centros) == 1:
                pts.append((x, centros[0]))
            continue
        prev = pts[-1][1]
        pred = prev if len(pts) < 2 else prev + (prev - pts[-2][1]) * (x - pts[-1][0]) / max(pts[-1][0] - pts[-2][0], 1)
        c = min(centros, key=lambda y: abs(y - pred))
        if abs(c - pred) <= salto:
            pts.append((x, c))
    return np.array(pts, dtype=float).reshape(-1, 2)
