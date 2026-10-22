#!/bin/bash
# SPDX-License-Identifier: GPL-3.0-or-later
# V1 of validacao/LEIAME.md: argon DBD of Stankov et al. 2020 (3 mm gap, 1 mm
# quartz barriers with eps 4.2, 24 kHz, U_a = -U0 sin(2 pi t/T), gamma 0.02,
# secondary electrons of 2 eV, energy equation, 300 K without heating,
# quasi-neutral initial state of 1e12 m-3). 800 records per period.
# Usage: remoto/validacao_stankov.sh PYTHON NAME:CELLS:P_MBAR:U0_V:PERIODS[:RTOL[:GAMMA[:EPS_R[:R_E[:CASE]]]]] ...
# R_E is the electron reflection coefficient (default 0) and CASE a folder of casos/ (default: the argon
# case at the repository root).
cd "$(dirname "$0")/.."
PY=${1:-python3}
shift
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
DIR=execucoes/validacao_stankov
mkdir -p $DIR
echo "start: $(date)"
python3 -c "import sys; sys.path.insert(0, 'prototipo'); from versao import VERSAO; print('dbd1d', VERSAO)"; git rev-parse HEAD 2>/dev/null
T=4.1666666666666665e-05
for c in "$@"; do
    IFS=: read -r nome n p u0 periodos rtol gama eps refl caso <<< "$c"
    rtol=${rtol:-1e-4}
    gama=${gama:-0.02}
    eps=${eps:-4.2}
    refl=${refl:-0}
    extra=""
    [ -n "$caso" ] && extra="--case casos/$caso"
    fim=$(python3 -c "print(repr($periodos * $T))")
    dt=$(python3 -c "print(repr($T / 800))")
    pa=$(python3 -c "print($p * 100.0)")
    ( /usr/bin/time -f "wall time: %e s" $PY prototipo/simula.py $DIR/$nome \
        --cells $n --clustering 3.0 --energy --rtol $rtol --electron-reflection $refl $extra \
        --gap 3e-3 --thickness 1e-3 --eps-dielectric $eps --gamma $gama --eps-sec 2 \
        --pressure $pa --t-gas 300 --waveform sine --period $T --v-peak -$u0 \
        --neutral --ne0 1e6 --t-end $fim --dt-output $dt ) > $DIR/$nome.log 2>&1 &
done
wait
echo "end: $(date)"
