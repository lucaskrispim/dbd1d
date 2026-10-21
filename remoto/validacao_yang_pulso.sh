#!/bin/bash
# SPDX-License-Identifier: GPL-3.0-or-later
# Energy and current of chosen pulses of a run of remoto/validacao_yang.sh:
# restarts from the state saved at the start of pulse n (t = (n-1) 10 us) and
# writes every 0.5 ns for 240 ns, with the same options (scripts/retoma_fina.py
# does the same for any run).
# Usage: remoto/validacao_yang_pulso.sh PYTHON ORIGIN CASE CELLS N1 N2 ...
#   ORIGIN: name of the run in execucoes/validacao_yang
cd "$(dirname "$0")/.."
PY=${1:-python3}
ORIGEM=$2
CASO=$3
N=$4
shift 4
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
DIR=execucoes/validacao_yang
echo "start: $(date)"
cat VERSAO 2>/dev/null || git rev-parse HEAD 2>/dev/null
for p in "$@"; do
    t0=$(python3 -c "print(repr(($p - 1) * 1e-5))")
    fim=$(python3 -c "print(repr(($p - 1) * 1e-5 + 2.4e-7))")
    ( /usr/bin/time -f "wall time: %e s" $PY prototipo/simula.py $DIR/${ORIGEM}_p$p --case casos/$CASO \
        --restart $DIR/$ORIGEM $t0 \
        --cells $N --clustering 3.0 --energy --heating \
        --gap 8e-3 --thickness 1.75e-3 --eps-dielectric 3.8 \
        --pressure 19998 --t-gas 300 --t-wall 300 \
        --waveform table --voltage-table validacao/yang2022/pulso50.txt --period 1e-5 \
        --neutral --ne0 1e6 --t-end $fim --dt-output 5e-10 ) > $DIR/${ORIGEM}_p$p.log 2>&1 &
done
wait
echo "end: $(date)"
