#!/bin/bash
# SPDX-License-Identifier: GPL-3.0-or-later
# Self-consistent coupled solutions of the N2 positive column by the root method
# (scripts/coluna_n2_raiz.py, 30/10/2026): the eight conditions, the two
# vibrational wall probabilities and the three steps of the coupling, plus the two
# sensitivity tests at 50 and 80 mA (wall of Gordiets et al.). The longest (530 Pa
# and the wider tube, full coupling) first; xargs keeps SIMULTANEOS in parallel.
# PARTE i/n: only the jobs whose position in the list is i modulo n (to split the
# list between machines). SUFIXO _ev: the modes with e-V from the excited levels
# (2/11/2026); the test limiar01 is left out then (it only touches v >= 24).
# Uso: remoto/raiz_coluna.sh SIMULTANEOS PYTHON ZDPLASKIN_DIR [PARTE [SUFIXO]]
cd "$(dirname "$0")/.."
SIM=${1:-16}
PY=${2:-python3}
export ZDPLASKIN_DIR=${3:-$ZDPLASKIN_DIR}
PARTE=${4:-0/1}
SUF=${5:-}
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
SAIDA=docs/coluna_positiva_n2/pendencias/raiz
mkdir -p $SAIDA
lista=$(mktemp)
for modo in completo deslocado alvo; do
    for id in B21_530Pa G95_80mA G95_30mA G95_15mA B21_230Pa L20_50mA L20_20mA L20_10mA; do
        for variante in base vib_gordiets; do
            echo "$id $variante $modo$SUF" >> $lista
        done
    done
done
echo "L20_50mA vib_gordiets meio$SUF" >> $lista
echo "G95_80mA vib_gordiets meio$SUF" >> $lista
if [ -z "$SUF" ]; then
    echo "L20_50mA vib_gordiets limiar01" >> $lista
    echo "G95_80mA vib_gordiets limiar01" >> $lista
fi
awk -v p="$PARTE" 'BEGIN{split(p, a, "/")} (NR - 1) % a[2] == a[1]' $lista > $lista.parte && mv $lista.parte $lista
echo "inicio: $(date), $(wc -l < $lista) jobs (parte $PARTE)"
git rev-parse HEAD 2>/dev/null || cat VERSAO 2>/dev/null
xargs -P $SIM -L 1 sh -c 'nome=$0_$1_$2; '"$PY"' scripts/coluna_n2_raiz.py $0 '"$SAIDA"' $1 $2 > '"$SAIDA"'/$nome.log 2>&1; echo "$nome: $?"' < $lista
rm -f $lista
echo "fim: $(date)"
