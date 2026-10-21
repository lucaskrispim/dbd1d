#!/bin/bash
# SPDX-License-Identifier: GPL-3.0-or-later
# V2 de validacao/LEIAME.md: tensão de ignição de Loffhagen et al. 2020 (argônio
# a 1 atm e 300 K, gap de 1 mm, borofloat de 2 mm com eps 4,6, 86,2 kHz,
# U_a = U0 sen(2 pi f t), gama 0,02, secundários com 2 eV, equação de energia).
# Gravação de 400 registros por período.
# Uso: remoto/validacao_loffhagen.sh PYTHON NOME:CELULAS:U0_V:PERIODOS[:RTOL] ...
cd "$(dirname "$0")/.."
PY=${1:-python3}
shift
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
DIR=execucoes/validacao_loffhagen
mkdir -p $DIR
echo "inicio: $(date)"
cat VERSAO 2>/dev/null || git rev-parse HEAD 2>/dev/null
T=1.1600928074245939e-05
for c in "$@"; do
    IFS=: read -r nome n u0 periodos rtol <<< "$c"
    rtol=${rtol:-1e-4}
    fim=$(python3 -c "print(repr($periodos * $T))")
    dt=$(python3 -c "print(repr($T / 400))")
    ( /usr/bin/time -f "tempo de parede: %e s" $PY prototipo/simula.py $DIR/$nome \
        --n-celulas $n --agrupamento 3.0 --energia --rtol $rtol \
        --gap 1e-3 --espessura 2e-3 --eps-dieletrico 4.6 --gama 0.02 --eps-sec 2 \
        --pressao 101325 --t-gas 300 --forma seno --periodo $T --v-pico $u0 \
        --neutro --ne0 1e6 --t-end $fim --dt-saida $dt ) > $DIR/$nome.log 2>&1 &
done
wait
echo "fim: $(date)"
