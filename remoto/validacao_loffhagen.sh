#!/bin/bash
# SPDX-License-Identifier: GPL-3.0-or-later
# V2 of validacao/LEIAME.md: ignition voltage of Loffhagen et al. 2020 (argon at
# 1 atm and 300 K, 1 mm gap, 2 mm borofloat barriers with eps 4.6, 86.2 kHz,
# U_a = U0 sin(2 pi f t), gamma 0.02, secondary electrons of 2 eV, energy
# equation). 400 records per period.
# Usage: remoto/validacao_loffhagen.sh PYTHON NAME:CELLS:U0_V:PERIODS[:RTOL] ...
cd "$(dirname "$0")/.."
PY=${1:-python3}
shift
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
DIR=execucoes/validacao_loffhagen
mkdir -p $DIR
echo "start: $(date)"
cat VERSAO 2>/dev/null || git rev-parse HEAD 2>/dev/null
T=1.1600928074245939e-05
for c in "$@"; do
    IFS=: read -r nome n u0 periodos rtol <<< "$c"
    rtol=${rtol:-1e-4}
    fim=$(python3 -c "print(repr($periodos * $T))")
    dt=$(python3 -c "print(repr($T / 400))")
    ( /usr/bin/time -f "wall time: %e s" $PY prototipo/simula.py $DIR/$nome \
        --cells $n --clustering 3.0 --energy --rtol $rtol \
        --gap 1e-3 --thickness 2e-3 --eps-dielectric 4.6 --gamma 0.02 --eps-sec 2 \
        --pressure 101325 --t-gas 300 --waveform sine --period $T --v-peak $u0 \
        --neutral --ne0 1e6 --t-end $fim --dt-output $dt ) > $DIR/$nome.log 2>&1 &
done
wait
echo "end: $(date)"
