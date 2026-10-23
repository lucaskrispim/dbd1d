#!/bin/bash
# SPDX-License-Identifier: GPL-3.0-or-later
# Every result of docs/coluna_positiva_n2 and the figures of the article on the N2
# positive column, from the rebuilt N2 chemistries (docs/coluna_positiva_n2/README.md).
# Needs ZDPlasKin and BOLSIG+ (ZDPLASKIN_DIR) and the cases xu2025, ar_n2_o2 and n2
# rebuilt as in casos/README.md. The coupled runs take 20 to 40 min per iteration.
#
# Usage: scripts/coluna_n2_todos.sh [PROCESSES]   (parallel runs, default 4)
set -euo pipefail
cd "$(dirname "$0")/.."
P=${1:-4}
D=docs/coluna_positiva_n2
IDS="L20_10mA L20_20mA L20_50mA B21_230Pa B21_530Pa G95_15mA G95_30mA G95_80mA"

# 1. the five N2 chemistries, and the cases derived from n2_amonia by the scripts
python3 ferramentas/casos_coluna.py
for c in n2_exemplo n2_amonia n2_tese n2_amonia_v45 n2_tese_v45; do
  python3 ferramentas/caso.py all casos/$c
done
python3 scripts/coluna_n2_superelastica.py casos
python3 scripts/coluna_n2_etapas.py caso

# 2. the analyses with the chemistries truncated at v = 8
mkdir -p $D/etapa4 $D/etapa5 $D/etapa6 $D/falta/acoplado lote
printf '%s\n' "coluna_n2.py $D/etapa4" "coluna_n2_en.py $D/etapa5" "coluna_n2_v3.py $D/etapa6" \
  "coluna_n2_parede.py $D/falta" "coluna_n2_convergencia.py $D/falta/convergencia.csv" \
  "coluna_n2_vazao.py $D/falta" "coluna_n2_superelastica.py en $D/falta" "coluna_n2_etapas.py en $D/falta" \
  "coluna_n2_sondas.py $D/falta/sondas.csv" | xargs -P "$P" -L 1 sh -c 'python3 scripts/$0 "$@"'

# 3. the ladder up to v = 45 with each set of wall probabilities (falta/escada_v45.csv)
for id in $IDS; do
  for v in base vib_gordiets meta_0.01 meta_0.1 N_x10 N_div10 vib_x10 vib_div10; do echo "n2_amonia_v45 $id $v"; done
  for v in base vib_gordiets meta_0.01; do echo "n2_tese_v45 $id $v"; done
done | xargs -P "$P" -L 1 sh -c 'python3 scripts/coluna_n2_lote.py $0 $1 $2 lote/$0_$1_$2.csv'
(echo "caso,id,variante,tv,dis,en"; for f in lote/*.csv; do tail -n +2 "$f"; done) > $D/falta/escada_v45.csv

# 4. the coupling, four iterations per chain (falta/acoplado/*.json): the three steps of the
# decomposition with the two vibrational wall probabilities, and the two tests of sensitivity
cadeia() {
  for k in 0 1 2 3; do python3 scripts/coluna_n2_acoplado.py "$1" $k $D/falta/acoplado "$2" "$3"; done
}
export -f cadeia
export D
{
  for m in alvo deslocado completo; do for v in base vib_gordiets; do for id in $IDS; do echo "$id $v $m"; done; done; done
  for id in G95_80mA L20_50mA; do for m in meio limiar01; do echo "$id vib_gordiets $m"; done; done
} | xargs -P "$P" -L 1 bash -c 'cadeia "$0" "$1" "$2"'

# 5. the figures of the article
python3 scripts/figuras_coluna_n2.py artigo_coluna_n2/figuras
