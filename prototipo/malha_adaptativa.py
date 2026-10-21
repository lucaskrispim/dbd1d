# SPDX-License-Identifier: GPL-3.0-or-later
"""Malha adaptativa em 1D com número fixo de células (malha móvel).

A malha nova sai por equidistribuição de uma densidade de células
    rho(x) = rho_base(x) + peso*rho_sol(x),
com rho_base a da malha de referência (a agrupada nas paredes, que garante as
bainhas mesmo sem gradientes) e rho_sol proporcional ao monitor, a soma dos
gradientes normalizados das grandezas escolhidas (ln n_e, ln n_i, campo). As
duas partes têm a mesma integral (m células), e a variação de rho entre células
vizinhas é limitada, o que limita a razão entre larguras vizinhas.

As densidades passam da malha velha para a nova por reconstrução linear com
inclinação limitada (minmod) e integrais de sobreposição: o total de cada
grandeza se conserva exatamente, a positividade se mantém e perfis lineares
são transferidos sem erro.
"""
import numpy as np


def centros(faces):
    return 0.5 * (faces[1:] + faces[:-1])


def monitor(faces, campos):
    """Soma, por célula, de |dq/dx|/(max q - min q) das grandezas em campos."""
    xc = centros(faces)
    total = np.zeros(len(xc))
    for q in campos:
        q = np.asarray(q, dtype=float)
        amplitude = q.max() - q.min()
        if amplitude <= 0.0:
            continue
        grad = np.gradient(q, xc)
        total += np.abs(grad) / amplitude
    return total


def faces_por_equidistribuicao(faces, mon, base, peso=1.0, razao_max=1.3, h_min=1.0e-7):
    """Faces novas (mesmo número de células) a partir do monitor mon nas células
    da malha faces, combinado com a densidade de células da malha base; nenhuma
    célula fica menor que h_min (m)."""
    m = len(faces) - 1
    h = np.diff(faces)
    xc = centros(faces)
    # densidade da malha base (células por metro) no centro de cada célula velha
    h_base = np.diff(base)
    k = np.clip(np.searchsorted(base, xc, side="right") - 1, 0, m - 1)
    rho = 1.0 / h_base[k]
    integral_mon = np.sum(mon * h)
    if integral_mon > 0.0:
        rho = rho + peso * m * mon / integral_mon
    # densidade máxima 1/h_min; como a soma das células é fixa, a escala final
    # reduz rho um pouco, então o teto leva uma folga
    # variação limitada por célula nova. Dentro de uma célula velha a densidade
    # é constante, e o tamanho das células novas salta na fronteira entre duas
    # velhas: ali a razão é no máximo razao_max. Onde uma célula velha contém só
    # uma fração n < 1 de célula nova, a queda permitida é razao_max**n, senão o
    # limite se acumularia dentro de uma única célula nova. n = rho*h*m/total.
    for _ in range(6):
        escala = m / np.sum(rho * h)
        n_novas = rho * h * escala
        for i in range(1, m):
            rho[i] = max(rho[i], rho[i - 1] * razao_max ** (-min(1.0, n_novas[i - 1])))
        for i in range(m - 2, -1, -1):
            rho[i] = max(rho[i], rho[i + 1] * razao_max ** (-min(1.0, n_novas[i + 1])))
    for _ in range(20):
        escala = m / np.sum(rho * h)
        excesso = rho * escala > 1.0 / h_min
        if not excesso.any():
            break
        rho[excesso] = 1.0 / (h_min * escala)
    acumulado = np.concatenate([[0.0], np.cumsum(rho * h)])
    alvo = np.linspace(0.0, acumulado[-1], m + 1)
    novas = np.interp(alvo, acumulado, faces)
    novas[0], novas[-1] = faces[0], faces[-1]
    return novas


def _inclinacoes(u, xc):
    """Inclinações minmod por célula (k, m)."""
    s = np.zeros_like(u)
    dx = np.diff(xc)
    d = np.diff(u, axis=1) / dx
    esq, dir_ = d[:, :-1], d[:, 1:]
    mm = np.where(esq * dir_ > 0.0, np.sign(esq) * np.minimum(np.abs(esq), np.abs(dir_)), 0.0)
    s[:, 1:-1] = mm
    # células das pontas: a inclinação do único lado; a limitação que preserva a
    # positividade nelas fica em transfere
    s[:, 0] = d[:, 0]
    s[:, -1] = d[:, -1]
    return s


def transfere(u, velhas, novas):
    """Médias por célula da malha nova a partir das da malha velha, u (k, m)."""
    u = np.atleast_2d(np.asarray(u, dtype=float))
    xc = centros(velhas)
    s = _inclinacoes(u, xc)
    # nas pontas, a reconstrução linear de um lado só não pode ficar negativa
    # dentro da célula
    for p in (0, -1):
        meia = 0.5 * (velhas[1] - velhas[0]) if p == 0 else 0.5 * (velhas[-1] - velhas[-2])
        excesso = np.abs(s[:, p]) * meia > u[:, p]
        s[excesso, p] = 0.0
    pontos = np.union1d(velhas, novas)
    a, b = pontos[:-1], pontos[1:]
    meio = 0.5 * (a + b)
    iv = np.clip(np.searchsorted(velhas, meio, side="right") - 1, 0, len(velhas) - 2)
    jn = np.clip(np.searchsorted(novas, meio, side="right") - 1, 0, len(novas) - 2)
    c = xc[iv]
    integral = u[:, iv] * (b - a) + s[:, iv] * 0.5 * ((b - c) ** 2 - (a - c) ** 2)
    total = np.zeros((u.shape[0], len(novas) - 1))
    for k in range(u.shape[0]):
        np.add.at(total[k], jn, integral[k])
    return total / np.diff(novas)


def transfere_energia(ne, n_eps, velhas, novas, ne_novo):
    """Densidade de energia na malha nova pela energia média ponderada pelos
    elétrons: eps_novo = int(n_eps)/int(n_e) sobre a célula nova, com médias
    constantes por célula velha. Transferir n_e e n_eps com reconstruções
    limitadas independentes criaria energias médias espúrias numa frente
    íngreme; assim eps_novo fica entre os valores vizinhos da malha velha. A
    energia total se conserva até a diferença entre esta média e a de n_e."""
    a = transfere_constante(np.vstack([np.maximum(ne, 0.0), np.maximum(n_eps, 0.0)]), velhas, novas)
    with np.errstate(invalid="ignore", divide="ignore"):
        eps = np.where(a[0] > 0.0, a[1] / a[0], 0.0)
    return np.maximum(ne_novo, 0.0) * eps


def transfere_constante(u, velhas, novas):
    """Transferência conservativa com valor constante por célula (primeira ordem)."""
    u = np.atleast_2d(np.asarray(u, dtype=float))
    pontos = np.union1d(velhas, novas)
    a, b = pontos[:-1], pontos[1:]
    meio = 0.5 * (a + b)
    iv = np.clip(np.searchsorted(velhas, meio, side="right") - 1, 0, len(velhas) - 2)
    jn = np.clip(np.searchsorted(novas, meio, side="right") - 1, 0, len(novas) - 2)
    total = np.zeros((u.shape[0], len(novas) - 1))
    for k in range(u.shape[0]):
        np.add.at(total[k], jn, u[k, iv] * (b - a))
    return total / np.diff(novas)
