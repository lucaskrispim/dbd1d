#!/bin/bash
# SPDX-License-Identifier: GPL-3.0-or-later
# Custo do jacobiano analítico com o potencial no estado (--jac dae, padrão)
# contra o de diferenças finitas do scipy (--jac fd), no mesmo trecho: os 8
# primeiros µs do argônio de Stankov et al. 2020 a 100 mbar com 50 células,
# que incluem a primeira ruptura (perto de 6 µs).
# Uso: validacao/compara_jacobiano.sh [PYTHON] [DIR]
cd "$(dirname "$0")/.."
PY=${1:-python3}
DIR=${2:-execucoes/jacobiano}
export OPENBLAS_NUM_THREADS=1
T=4.1666666666666665e-05
for jac in dae fd; do
    /usr/bin/time -f "$jac: tempo de parede %e s" $PY prototipo/simula.py $DIR/$jac --jac $jac \
        --n-celulas 50 --agrupamento 3.0 --energia --gap 3e-3 --espessura 1e-3 --eps-dieletrico 4.2 \
        --gama 0.02 --eps-sec 2 --pressao 1e4 --t-gas 300 --forma seno --periodo $T --v-pico -1800 \
        --neutro --ne0 1e6 --t-end 8e-6 --dt-saida 1e-7 > $DIR.$jac.log 2>&1
    tail -1 $DIR.$jac.log
done
