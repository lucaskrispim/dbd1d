#!/bin/bash
# SPDX-License-Identifier: GPL-3.0-or-later
# Energia e corrente de pulsos escolhidos de uma execução de
# remoto/validacao_yang.sh: retoma do estado gravado no início do pulso n
# (t = (n-1) 10 µs) e grava a cada 0,5 ns por 240 ns, com as mesmas opções.
# Uso: remoto/validacao_yang_pulso.sh PYTHON ORIGEM CASO CELULAS N1 N2 ...
#   ORIGEM: nome da execução em execucoes/validacao_yang
cd "$(dirname "$0")/.."
PY=${1:-python3}
ORIGEM=$2
CASO=$3
N=$4
shift 4
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
DIR=execucoes/validacao_yang
echo "inicio: $(date)"
cat VERSAO 2>/dev/null || git rev-parse HEAD 2>/dev/null
for p in "$@"; do
    t0=$(python3 -c "print(repr(($p - 1) * 1e-5))")
    fim=$(python3 -c "print(repr(($p - 1) * 1e-5 + 2.4e-7))")
    ( /usr/bin/time -f "tempo de parede: %e s" $PY prototipo/simula.py $DIR/${ORIGEM}_p$p --caso casos/$CASO \
        --retoma $DIR/$ORIGEM $t0 \
        --n-celulas $N --agrupamento 3.0 --energia --aquecimento \
        --gap 8e-3 --espessura 1.75e-3 --eps-dieletrico 3.8 \
        --pressao 19998 --t-gas 300 --t-parede 300 \
        --forma tabela --tensao-tabela validacao/yang2022/pulso50.txt --periodo 1e-5 \
        --neutro --ne0 1e6 --t-end $fim --dt-saida 5e-10 ) > $DIR/${ORIGEM}_p$p.log 2>&1 &
done
wait
echo "fim: $(date)"
