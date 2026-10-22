#!/bin/bash
# SPDX-License-Identifier: GPL-3.0-or-later
# Energy and current of chosen pulses of a run of remoto/validacao_takashima.sh:
# restarts from the state saved at the start of pulse n (t = (n-1) 25 us) and
# writes every 0.5 ns for 240 ns, with the same options (scripts/retoma_fina.py
# does the same for any run).
# Usage: remoto/validacao_takashima_pulso.sh PYTHON ORIGIN P_TORR CELLS EPS_R N1 N2 ...
#   ORIGIN: name of the run in execucoes/validacao_takashima
cd "$(dirname "$0")/.."
PY=${1:-python3}
ORIGEM=$2
P_TORR=$3
PA=$(python3 -c "print($P_TORR * 133.322)")
N=$4
EPS=$5
shift 5
PULSO=validacao/takashima2012/pulso_cpt.txt
[ "$EPS" != "3.8" ] && PULSO=validacao/takashima2012/pulso_cpt_eps$EPS.txt
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
DIR=execucoes/validacao_takashima
echo "start: $(date)"
python3 -c "import sys; sys.path.insert(0, 'prototipo'); from versao import VERSAO; print('dbd1d', VERSAO)"; git rev-parse HEAD 2>/dev/null
for p in "$@"; do
    t0=$(python3 -c "print(repr(($p - 1) * 2.5e-5))")
    fim=$(python3 -c "print(repr(($p - 1) * 2.5e-5 + 2.4e-7))")
    ( /usr/bin/time -f "wall time: %e s" $PY prototipo/simula.py $DIR/${ORIGEM}_p$p --case casos/ar_n2_o2 \
        --restart $DIR/$ORIGEM $t0 \
        --cells $N --clustering 3.0 --energy --heating \
        --gap 1e-2 --thickness 1.75e-3 --eps-dielectric $EPS \
        --pressure $PA --t-gas 300 --t-wall 300 \
        --waveform table --voltage-table $PULSO --period 2.5e-5 \
        --neutral --ne0 1e6 --t-end $fim --dt-output 5e-10 ) > $DIR/${ORIGEM}_p$p.log 2>&1 &
done
wait
echo "end: $(date)"
