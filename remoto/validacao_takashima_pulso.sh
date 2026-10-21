#!/bin/bash
# SPDX-License-Identifier: GPL-3.0-or-later
# Energia e corrente de pulsos escolhidos de uma execução de
# remoto/validacao_takashima.sh: retoma do estado gravado no início do pulso n
# (t = (n-1) 10 µs) e grava a cada 0,5 ns por 240 ns, com as mesmas opções.
# Uso: remoto/validacao_takashima_pulso.sh PYTHON ORIGEM CASO CELULAS N1 N2 ...
#   ORIGEM: nome da execução em execucoes/validacao_takashima
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
echo "inicio: $(date)"
cat VERSAO 2>/dev/null || git rev-parse HEAD 2>/dev/null
for p in "$@"; do
    t0=$(python3 -c "print(repr(($p - 1) * 2.5e-5))")
    fim=$(python3 -c "print(repr(($p - 1) * 2.5e-5 + 2.4e-7))")
    ( /usr/bin/time -f "tempo de parede: %e s" $PY prototipo/simula.py $DIR/${ORIGEM}_p$p --caso casos/ar_n2_o2 \
        --retoma $DIR/$ORIGEM $t0 \
        --n-celulas $N --agrupamento 3.0 --energia --aquecimento \
        --gap 1e-2 --espessura 1.75e-3 --eps-dieletrico $EPS \
        --pressao $PA --t-gas 300 --t-parede 300 \
        --forma tabela --tensao-tabela $PULSO --periodo 2.5e-5 \
        --neutro --ne0 1e6 --t-end $fim --dt-saida 5e-10 ) > $DIR/${ORIGEM}_p$p.log 2>&1 &
done
wait
echo "fim: $(date)"
