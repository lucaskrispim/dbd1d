# SPDX-License-Identifier: GPL-3.0-or-later
"""Retomada fina de pulsos escolhidos de uma execução: retoma do estado
gravado no início do pulso n, (n - 1) períodos, e grava a cada DT por JANELA,
com as mesmas opções da execução de origem, lidas do parametros.json dela.
Serve a qualquer execução, inclusive às de sensibilidade, que mudam espessura,
permissividade ou malha (os remoto/validacao_*_pulso.sh têm essas opções fixas).

As retomadas rodam em paralelo, uma por pulso, em ORIGEM_pN com o log em
ORIGEM_pN.log.

Uso: python3 scripts/retoma_fina.py ORIGEM N1 [N2 ...] [--janela 2.4e-7] [--dt 5e-10] [--python PY]
"""
import argparse
import json
import os
import pathlib
import subprocess
import sys

RAIZ = pathlib.Path(__file__).resolve().parent.parent

# as que a retomada troca; as outras se repetem como estão
TROCADAS = ("dir", "retoma", "t_end", "dt_saida")


def inicio_do_pulso(n, periodo):
    return (n - 1) * periodo


def linha_de_comando(argumentos, destino, t0, t_end, dt_saida):
    """Argumentos do simula.py que repetem os da origem e retomam em t0."""
    linha = [str(destino)]
    for k, v in argumentos.items():
        if k in TROCADAS or v is None or v is False:
            continue
        opcao = "--" + k.replace("_", "-")
        linha += [opcao] if v is True else [opcao, repr(v) if isinstance(v, float) else str(v)]
    return linha + ["--retoma", argumentos["dir"], repr(t0), "--t-end", repr(t_end), "--dt-saida", repr(dt_saida)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("origem", type=pathlib.Path)
    p.add_argument("pulsos", type=int, nargs="+")
    p.add_argument("--janela", type=float, default=2.4e-7)
    p.add_argument("--dt", type=float, default=5e-10)
    p.add_argument("--python", default=sys.executable)
    a = p.parse_args()
    arg = json.loads((a.origem / "parametros.json").read_text())["argumentos"]
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    procs = []
    for n in a.pulsos:
        t0 = inicio_do_pulso(n, arg["periodo"])
        destino = a.origem.parent / f"{a.origem.name}_p{n}"
        cmd = ["/usr/bin/time", "-f", "tempo de parede: %e s", a.python, str(RAIZ / "prototipo" / "simula.py")] + \
            linha_de_comando(arg, destino, t0, t0 + a.janela, a.dt)
        log = open(f"{destino}.log", "w")
        procs.append(subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, cwd=RAIZ))
    sys.exit(max(pr.wait() for pr in procs))


if __name__ == "__main__":
    main()
