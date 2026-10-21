#!/bin/bash
# SPDX-License-Identifier: GPL-3.0-or-later
# V3 de validacao/LEIAME.md: rajada de pulsos de nanossegundos de Yang et al.
# 2022 (150 Torr, gap de 8 mm, quartzo de 1,75 mm com eps 3,8, pulso 50 medido
# repetido a 100 kHz, equação de energia, aquecimento com paredes a 300 K).
# Gravação a cada 1 µs (dez por período); a energia por pulso sai de retomadas
# curtas (remoto/validacao_yang_pulso.sh).
# Uso: remoto/validacao_yang.sh PYTHON NOME:CELULAS:CASO:PULSOS[:RTOL[:NE0[:ESPESSURA[:EPS_R]]]] ...
cd "$(dirname "$0")/.."
PY=${1:-python3}
shift
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
DIR=execucoes/validacao_yang
mkdir -p $DIR
echo "inicio: $(date)"
cat VERSAO 2>/dev/null || git rev-parse HEAD 2>/dev/null
for c in "$@"; do
    IFS=: read -r nome n caso pulsos rtol ne0 esp eps <<< "$c"
    rtol=${rtol:-1e-4}
    ne0=${ne0:-1e6}
    esp=${esp:-1.75e-3}
    eps=${eps:-3.8}
    fim=$(python3 -c "print(repr($pulsos * 1e-5))")
    ( /usr/bin/time -f "tempo de parede: %e s" $PY prototipo/simula.py $DIR/$nome --caso casos/$caso \
        --n-celulas $n --agrupamento 3.0 --energia --aquecimento --rtol $rtol \
        --gap 8e-3 --espessura $esp --eps-dieletrico $eps \
        --pressao 19998 --t-gas 300 --t-parede 300 \
        --forma tabela --tensao-tabela validacao/yang2022/pulso50.txt --periodo 1e-5 \
        --neutro --ne0 $ne0 --t-end $fim --dt-saida 1e-6 ) > $DIR/$nome.log 2>&1 &
done
wait
echo "fim: $(date)"
