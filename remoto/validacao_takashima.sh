#!/bin/bash
# SPDX-License-Identifier: GPL-3.0-or-later
# V5 de validacao/LEIAME.md: pulsos de nanossegundos em ar seco de Takashima et
# al. 2012, gerador CPT a 40 kHz (casos/ar_n2_o2: N2 78 %, O2 21 %, Ar 1 %),
# canal de 10 mm com paredes de quartzo de 1,75 mm (eps 3,8), equação de
# energia, aquecimento com paredes a 300 K. Gravação a cada 1 µs; a energia do
# último pulso sai de uma retomada fina (remoto/validacao_takashima_pulso.sh).
# Com EPS_R diferente de 3,8, o pulso é validacao/takashima2012/pulso_cpt_epsEPS_R.txt
# (a tensão sai do campo aplicado vezes d + 2 l/ε).
# Uso: remoto/validacao_takashima.sh PYTHON NOME:CELULAS:P_TORR:PULSOS[:EPS_R] ...
cd "$(dirname "$0")/.."
PY=${1:-python3}
shift
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
DIR=execucoes/validacao_takashima
mkdir -p $DIR
echo "inicio: $(date)"
cat VERSAO 2>/dev/null || git rev-parse HEAD 2>/dev/null
for c in "$@"; do
    IFS=: read -r nome n p pulsos eps <<< "$c"
    eps=${eps:-3.8}
    pulso=validacao/takashima2012/pulso_cpt.txt
    [ "$eps" != "3.8" ] && pulso=validacao/takashima2012/pulso_cpt_eps$eps.txt
    fim=$(python3 -c "print(repr($pulsos * 2.5e-5))")
    pa=$(python3 -c "print($p * 133.322)")
    ( /usr/bin/time -f "tempo de parede: %e s" $PY prototipo/simula.py $DIR/$nome --caso casos/ar_n2_o2 \
        --n-celulas $n --agrupamento 3.0 --energia --aquecimento \
        --gap 1e-2 --espessura 1.75e-3 --eps-dieletrico $eps \
        --pressao $pa --t-gas 300 --t-parede 300 \
        --forma tabela --tensao-tabela $pulso --periodo 2.5e-5 \
        --neutro --ne0 1e6 --t-end $fim --dt-saida 1e-6 ) > $DIR/$nome.log 2>&1 &
done
wait
echo "fim: $(date)"
