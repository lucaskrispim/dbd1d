#!/bin/bash
# Quick start: three periods of the argon DBD of Stankov et al. 2020 (3 mm gap,
# 1 mm quartz barriers, 100 mbar, 1.8 kV at 24 kHz) with 50 cells, then the
# discharge current against the measured one and the power against the
# measurement, in the third period. About three minutes on a laptop.
set -e
cd "$(dirname "$0")/.."
export OPENBLAS_NUM_THREADS=1
T=4.1666666666666665e-05
python3 prototipo/simula.py execucoes/quickstart --n-celulas 50 --agrupamento 3.0 --energia \
    --gap 3e-3 --espessura 1e-3 --eps-dieletrico 4.2 --gama 0.02 --eps-sec 2 \
    --pressao 1e4 --t-gas 300 --forma seno --periodo $T --v-pico -1800 \
    --neutro --ne0 1e6 --t-end $(python3 -c "print(repr(3 * $T))") --dt-saida $(python3 -c "print(repr($T / 400))")
python3 scripts/compara_stankov.py execucoes/quickstart --figura execucoes/quickstart.png
echo "figure: execucoes/quickstart.png (third period; the validation runs ten)"
