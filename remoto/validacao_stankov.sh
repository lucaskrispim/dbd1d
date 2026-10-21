#!/bin/bash
# SPDX-License-Identifier: GPL-3.0-or-later
# V1 de validacao/LEIAME.md: DBD de argônio de Stankov et al. 2020 (gap de 3 mm,
# quartzo de 1 mm com eps 4,2, 24 kHz, U_a = -U0 sen(2 pi t/T), gama 0,02,
# secundários com 2 eV, equação de energia, 300 K sem aquecimento, estado
# inicial quase neutro de 1e12 m-3). Gravação de 800 registros por período.
# Uso: remoto/validacao_stankov.sh PYTHON NOME:CELULAS:P_MBAR:U0_V:PERIODOS[:RTOL[:GAMA[:EPS_R]]] ...
cd "$(dirname "$0")/.."
PY=${1:-python3}
shift
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
DIR=execucoes/validacao_stankov
mkdir -p $DIR
echo "inicio: $(date)"
cat VERSAO 2>/dev/null || git rev-parse HEAD 2>/dev/null
T=4.1666666666666665e-05
for c in "$@"; do
    IFS=: read -r nome n p u0 periodos rtol gama eps <<< "$c"
    rtol=${rtol:-1e-4}
    gama=${gama:-0.02}
    eps=${eps:-4.2}
    fim=$(python3 -c "print(repr($periodos * $T))")
    dt=$(python3 -c "print(repr($T / 800))")
    pa=$(python3 -c "print($p * 100.0)")
    ( /usr/bin/time -f "tempo de parede: %e s" $PY prototipo/simula.py $DIR/$nome \
        --n-celulas $n --agrupamento 3.0 --energia --rtol $rtol \
        --gap 3e-3 --espessura 1e-3 --eps-dieletrico $eps --gama $gama --eps-sec 2 \
        --pressao $pa --t-gas 300 --forma seno --periodo $T --v-pico -$u0 \
        --neutro --ne0 1e6 --t-end $fim --dt-saida $dt ) > $DIR/$nome.log 2>&1 &
done
wait
echo "fim: $(date)"
