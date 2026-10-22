# Reproducing the results of the article

`python3 validacao/curta.py` checks reduced versions of the comparisons in a few minutes (see the
README). The full runs below give the numbers of the article.

Every run of the article is listed below with the command that produces it. The scripts write to
`execucoes/validacao_*` and start the runs in parallel, one process per run; the times in the
performance table of the article are per core. The figures are then drawn by

```
python3 artigo/figuras.py execucoes
```

which reads `execucoes/validacao_stankov`, `validacao_loffhagen`, `validacao_takashima` and
`validacao_yang` and writes `artigo/figuras/*.pdf`. The numbers of the text come from the
comparison scripts named below. `PY` is the Python of the run (`python3` or a virtual
environment). The cases `ar_n2_o2`, `yang2022_*` and their variants need the files described in
`casos/LEIAME.md`; the argon cases run as distributed.

## Argon, sine excitation (Stankov et al. 2020)

Format: `NAME:CELLS:P_MBAR:U0_V:PERIODS[:RTOL[:GAMMA[:EPS_R[:R_E[:CASE]]]]]`.

```
remoto/validacao_stankov.sh PY S100_n100:100:100:1800:10 S300_n100:100:300:2200:10 \
    S500_n100:100:500:2700:10 S650_n100:100:650:3400:10
remoto/validacao_stankov.sh PY S100_n200:200:100:1800:10 S100_n400:400:100:1800:10 \
    S100_n100_r5:100:100:1800:10:1e-5
remoto/validacao_stankov.sh PY G0.005:100:100:1800:10:1e-4:0.005 G0.01:100:100:1800:10:1e-4:0.01 \
    G0.04:100:100:1800:10:1e-4:0.04 G0.08:100:100:1800:10:1e-4:0.08
remoto/validacao_stankov.sh PY E3.3:100:100:1800:10:1e-4:0.02:3.3 E3.75:100:100:1800:10:1e-4:0.02:3.75 \
    E4.65:100:100:1800:10:1e-4:0.02:4.65 E5.1:100:100:1800:10:1e-4:0.02:5.1
remoto/validacao_stankov.sh PY J100:100:100:1800:10:1e-4:0.02:4.2:0:argonio_jovanovic \
    J300:100:300:2200:10:1e-4:0.02:4.2:0:argonio_jovanovic \
    J500:100:500:2700:10:1e-4:0.02:4.2:0:argonio_jovanovic \
    J650:100:650:3400:10:1e-4:0.02:4.2:0:argonio_jovanovic
remoto/validacao_stankov.sh PY R100_r07:100:100:1800:10:1e-4:0.02:4.2:0.7 R300_r07:100:300:2200:10:1e-4:0.02:4.2:0.7 \
    R500_r07:100:500:2700:10:1e-4:0.02:4.2:0.7 R650_r07:100:650:3400:10:1e-4:0.02:4.2:0.7
remoto/validacao_stankov.sh PY JC100:100:100:1800:10:1e-4:0.02:4.2:0:argonio_jovanovic_ci \
    JT100:100:100:1800:10:1e-4:0.02:4.2:0:argonio_jovanovic_tc
python3 scripts/compara_stankov.py execucoes/validacao_stankov/S*_n100
python3 scripts/detalha_stankov.py execucoes/validacao_stankov/S100_n100
python3 scripts/fluxos.py --species E execucoes/validacao_stankov/J100
```

## Ignition of argon at 1 atm (Loffhagen et al. 2020)

Format: `NAME:CELLS:U0_V:PERIODS[:RTOL[:GAMMA[:NE0[:CASE]]]]`.

```
a=""; for u in 2000 2250 2500 2750 2800 2850 2900 2950 3000 3250 3500 3750 4000; do a="$a L$u:100:$u:10"; done
remoto/validacao_loffhagen.sh PY $a
for g in 0.01 0.08; do b=""; for u in $(seq 2400 100 3600); do b="$b LG${g}_$u:100:$u:10:1e-4:$g"; done
    remoto/validacao_loffhagen.sh PY $b; done
c=""; for u in $(seq 2600 100 3200); do c="$c LJ_$u:100:$u:10:1e-4:0.02:1e6:argonio_jovanovic"; done
remoto/validacao_loffhagen.sh PY $c
for ne in 1e4 1e8; do d=""; for u in 2700 2800 2900 3000; do d="$d LN${ne}_$u:100:$u:10:1e-4:0.02:$ne"; done
    remoto/validacao_loffhagen.sh PY $d; done
python3 scripts/compara_loffhagen.py execucoes/validacao_loffhagen/L[0-9]*[0-9]
```

## Nanosecond pulses in dry air (Takashima et al. 2012)

Format: `NAME:CELLS:P_TORR:PULSES[:EPS_R]`; the energy of the fifth pulse comes from a fine
restart of it.

```
remoto/validacao_takashima.sh PY T20:50:20:5 T40:50:40:5 T60:50:60:5 T100:50:100:5 T200:50:200:5 \
    T250:50:250:5 T275:50:275:5 T300:50:300:5
remoto/validacao_takashima.sh PY TS60_eps3.5:50:60:5:3.5 TS60_eps4.2:50:60:5:4.2 \
    TS200_eps3.5:50:200:5:3.5 TS200_eps4.2:50:200:5:4.2 TS60_n100:100:60:5 TS200_n100:100:200:5
for r in T20 T40 T60 T100 T200 T250 T275 T300 TS60_eps3.5 TS60_eps4.2 TS200_eps3.5 TS200_eps4.2 \
         TS60_n100 TS200_n100; do python3 scripts/retoma_fina.py execucoes/validacao_takashima/$r 5; done
python3 scripts/compara_takashima.py execucoes/validacao_takashima
```

## Bursts of nanosecond pulses in N2 and N2-H2 (Yang et al. 2022)

Format: `NAME:CELLS:CASE:PULSES[:RTOL[:NE0[:THICKNESS[:EPS_R]]]]`. The article uses the first 40
pulses of the 150-pulse runs, which take three to four and a half days, and `artigo/figuras.py`
shows only those pulses (40 in N2 and 25 with 1% H2). The variants of the mechanism were compared
with the base runs after pulse 6.

```
remoto/validacao_yang.sh PY YN2_n50:50:yang2022_N2:150 YH2_n50:50:yang2022_1pcH2:150
remoto/validacao_yang.sh PY YN2_n75:75:yang2022_N2:5 C_YN2:100:yang2022_N2:2 C_YN2_n200:200:yang2022_N2:2 \
    YN2_r5:50:yang2022_N2:2:1e-5
remoto/validacao_yang.sh PY YS_e1.5:50:yang2022_N2:5:1e-4:1e6:1.5e-3 YS_e2.0:50:yang2022_N2:5:1e-4:1e6:2.0e-3 \
    YS_eps3.5:50:yang2022_N2:5:1e-4:1e6:1.75e-3:3.5 YS_eps4.2:50:yang2022_N2:5:1e-4:1e6:1.75e-3:4.2
remoto/validacao_yang.sh PY YP_N2:50:yang2022_N2_pool50:6 YP_H2:50:yang2022_1pcH2_pool50:6 \
    YH05:50:yang2022_1pcH2_h05:6 YPH05:50:yang2022_1pcH2_pool50_h05:6 YC_n50:50:yang2022_N2_calor:5
python3 scripts/retoma_fina.py execucoes/validacao_yang/YN2_n50 2 5 10 20
python3 scripts/retoma_fina.py execucoes/validacao_yang/YH2_n50 2 5 10 20
python3 scripts/compara_yang.py execucoes/validacao_yang/YN2_n50 execucoes/validacao_yang/YN2_n50_p{2,5,10,20}
python3 scripts/compara_yang.py execucoes/validacao_yang/YH2_n50 execucoes/validacao_yang/YH2_n50_p{2,5,10,20}
```

## Verification and swarm data

```
python3 validacao/townsend.py
python3 scripts/compara_enxame.py
validacao/compara_jacobiano.sh
python3 validacao/desempenho.py execucoes
```
