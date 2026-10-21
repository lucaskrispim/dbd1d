#!/bin/bash
# SPDX-License-Identifier: GPL-3.0-or-later
# V3 of validacao/LEIAME.md: burst of nanosecond pulses of Yang et al. 2022 (150
# Torr, 8 mm gap, 1.75 mm quartz with eps 3.8, measured pulse 50 repeated at 100
# kHz, energy equation, heating with walls at 300 K). Records every 1 us (ten
# per period); the energy per pulse comes from short restarts
# (scripts/retoma_fina.py).
# Usage: remoto/validacao_yang.sh PYTHON NAME:CELLS:CASE:PULSES[:RTOL[:NE0[:THICKNESS[:EPS_R]]]] ...
cd "$(dirname "$0")/.."
PY=${1:-python3}
shift
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
DIR=execucoes/validacao_yang
mkdir -p $DIR
echo "start: $(date)"
cat VERSAO 2>/dev/null || git rev-parse HEAD 2>/dev/null
for c in "$@"; do
    IFS=: read -r nome n caso pulsos rtol ne0 esp eps <<< "$c"
    rtol=${rtol:-1e-4}
    ne0=${ne0:-1e6}
    esp=${esp:-1.75e-3}
    eps=${eps:-3.8}
    fim=$(python3 -c "print(repr($pulsos * 1e-5))")
    ( /usr/bin/time -f "wall time: %e s" $PY prototipo/simula.py $DIR/$nome --case casos/$caso \
        --cells $n --clustering 3.0 --energy --heating --rtol $rtol \
        --gap 8e-3 --thickness $esp --eps-dielectric $eps \
        --pressure 19998 --t-gas 300 --t-wall 300 \
        --waveform table --voltage-table validacao/yang2022/pulso50.txt --period 1e-5 \
        --neutral --ne0 $ne0 --t-end $fim --dt-output 1e-6 ) > $DIR/$nome.log 2>&1 &
done
wait
echo "end: $(date)"
