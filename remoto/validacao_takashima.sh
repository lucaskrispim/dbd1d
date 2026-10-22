#!/bin/bash
# SPDX-License-Identifier: GPL-3.0-or-later
# V5 of validacao/LEIAME.md: nanosecond pulses in dry air of Takashima et al.
# 2012, CPT generator at 40 kHz (casos/ar_n2_o2: N2 78%, O2 21%, Ar 1%), 10 mm
# channel with 1.75 mm quartz walls (eps 3.8), energy equation, heating with
# walls at 300 K. Records every 1 us; the energy of the last pulse comes from a
# fine restart (scripts/retoma_fina.py). With EPS_R other than 3.8, the pulse is
# validacao/takashima2012/pulso_cpt_epsEPS_R.txt (the voltage is the applied
# field times d + 2 l/eps).
# Usage: remoto/validacao_takashima.sh PYTHON NAME:CELLS:P_TORR:PULSES[:EPS_R] ...
cd "$(dirname "$0")/.."
PY=${1:-python3}
shift
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
DIR=execucoes/validacao_takashima
mkdir -p $DIR
echo "start: $(date)"
python3 -c "import sys; sys.path.insert(0, 'prototipo'); from versao import VERSAO; print('dbd1d', VERSAO)"; git rev-parse HEAD 2>/dev/null
for c in "$@"; do
    IFS=: read -r nome n p pulsos eps <<< "$c"
    eps=${eps:-3.8}
    pulso=validacao/takashima2012/pulso_cpt.txt
    [ "$eps" != "3.8" ] && pulso=validacao/takashima2012/pulso_cpt_eps$eps.txt
    fim=$(python3 -c "print(repr($pulsos * 2.5e-5))")
    pa=$(python3 -c "print($p * 133.322)")
    ( /usr/bin/time -f "wall time: %e s" $PY prototipo/simula.py $DIR/$nome --case casos/ar_n2_o2 \
        --cells $n --clustering 3.0 --energy --heating \
        --gap 1e-2 --thickness 1.75e-3 --eps-dielectric $eps \
        --pressure $pa --t-gas 300 --t-wall 300 \
        --waveform table --voltage-table $pulso --period 2.5e-5 \
        --neutral --ne0 1e6 --t-end $fim --dt-output 1e-6 ) > $DIR/$nome.log 2>&1 &
done
wait
echo "end: $(date)"
