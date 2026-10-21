#!/bin/bash
# SPDX-License-Identifier: GPL-3.0-or-later
# Cost of the structured Jacobian with the potential in the state (--jac dae,
# default) against scipy's finite-difference Jacobian (--jac fd), over the same
# stretch: the first 8 us of the argon case of Stankov et al. 2020 at 100 mbar
# with 50 cells, which include the first breakdown (near 6 us).
# Usage: validacao/compara_jacobiano.sh [PYTHON] [DIR]
cd "$(dirname "$0")/.."
PY=${1:-python3}
DIR=${2:-execucoes/jacobiano}
export OPENBLAS_NUM_THREADS=1
T=4.1666666666666665e-05
for jac in dae fd; do
    /usr/bin/time -f "$jac: wall time %e s" $PY prototipo/simula.py $DIR/$jac --jac $jac \
        --cells 50 --clustering 3.0 --energy --gap 3e-3 --thickness 1e-3 --eps-dielectric 4.2 \
        --gamma 0.02 --eps-sec 2 --pressure 1e4 --t-gas 300 --waveform sine --period $T --v-peak -1800 \
        --neutral --ne0 1e6 --t-end 8e-6 --dt-output 1e-7 > $DIR.$jac.log 2>&1
    tail -1 $DIR.$jac.log
done
