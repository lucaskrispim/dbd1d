#!/bin/bash
# SPDX-License-Identifier: GPL-3.0-or-later
# Quick start: three periods of the argon DBD of Stankov et al. 2020 (3 mm gap,
# 1 mm quartz barriers, 100 mbar, 1.8 kV at 24 kHz) with 50 cells, then the
# discharge current against the measured one and the power against the
# measurement, in the third period. About three minutes on a laptop.
set -e
cd "$(dirname "$0")/.."
export OPENBLAS_NUM_THREADS=1
T=4.1666666666666665e-05
python3 dbd1d/simula.py execucoes/quickstart --cells 50 --clustering 3.0 --energy \
    --gap 3e-3 --thickness 1e-3 --eps-dielectric 4.2 --gamma 0.02 --eps-sec 2 \
    --pressure 1e4 --t-gas 300 --waveform sine --period $T --v-peak -1800 \
    --neutral --ne0 1e6 --t-end $(python3 -c "print(repr(3 * $T))") --dt-output $(python3 -c "print(repr($T / 400))")
python3 scripts/compara_stankov.py execucoes/quickstart --figure execucoes/quickstart.png
echo "figure: execucoes/quickstart.png (third period; the validation runs ten)"
