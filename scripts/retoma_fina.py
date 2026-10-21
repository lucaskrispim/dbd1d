# SPDX-License-Identifier: GPL-3.0-or-later
"""Fine restart of chosen pulses of a run: restarts from the state saved at the
start of pulse n, (n - 1) periods, and writes every DT during WINDOW, with the
same options as the origin run, read from its parametros.json. Works for any
run, including the sensitivity runs, which change thickness, permittivity or mesh.

The restarts run in parallel, one per pulse, in ORIGIN_pN with the log in
ORIGIN_pN.log.

Usage: python3 scripts/retoma_fina.py ORIGIN N1 [N2 ...] [--window 2.4e-7] [--dt 5e-10] [--python PY]
"""
import argparse
import json
import os
import pathlib
import subprocess
import sys

RAIZ = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "prototipo"))

# the ones the restart changes; the others are repeated as they are
TROCADAS = ("dir", "retoma", "t_end", "dt_saida")


def inicio_do_pulso(n, periodo):
    return (n - 1) * periodo


def linha_de_comando(argumentos, destino, t0, t_end, dt_saida):
    """Arguments of simula.py that repeat those of the origin and restart at t0,
    with the English option names."""
    from simula import NOMES
    linha = [str(destino)]
    for k, v in argumentos.items():
        if k in TROCADAS or v is None or v is False:
            continue
        opcao = "--" + NOMES.get(k, k.replace("_", "-"))
        linha += [opcao] if v is True else [opcao, repr(v) if isinstance(v, float) else str(v)]
    return linha + ["--restart", argumentos["dir"], repr(t0), "--t-end", repr(t_end), "--dt-output", repr(dt_saida)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("origem", type=pathlib.Path)
    p.add_argument("pulsos", type=int, nargs="+")
    p.add_argument("--window", "--janela", dest="janela", type=float, default=2.4e-7)
    p.add_argument("--dt", type=float, default=5e-10)
    p.add_argument("--python", default=sys.executable)
    a = p.parse_args()
    arg = json.loads((a.origem / "parametros.json").read_text())["argumentos"]
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    procs = []
    for n in a.pulsos:
        t0 = inicio_do_pulso(n, arg["periodo"])
        destino = a.origem.parent / f"{a.origem.name}_p{n}"
        cmd = ["/usr/bin/time", "-f", "wall time: %e s", a.python, str(RAIZ / "prototipo" / "simula.py")] + \
            linha_de_comando(arg, destino, t0, t0 + a.janela, a.dt)
        log = open(f"{destino}.log", "w")
        procs.append(subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, cwd=RAIZ))
    sys.exit(max(pr.wait() for pr in procs))


if __name__ == "__main__":
    main()
