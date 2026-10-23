# N2 positive column

Data, scripts and results of the article on the N2 positive column: the N2 parts of three
chemistries (one of them from a mechanism used for plasma ammonia synthesis) tested against eight
measured conditions of flowing DC discharges in pure N2, with the zero-dimensional model of
`dbd1d/coluna.py`.

## Measured conditions

`condicoes.csv` holds the five conditions of Levaton et al. 2020 (Phys. Plasmas 27, 063503,
table VIII) and 2021 (Braz. J. Phys. 51, 75, table 1): tube radius 0.3 cm, 230 and 530 Pa, 10 to
60 mA. The three pure N2 points of Gordiets et al. 1995 (IEEE Trans. Plasma Sci. 23, 750; tube
radius 0.8 cm, 267 Pa, 15 to 80 mA) are in `docs/literatura/dados/gordiets1995/n2_puro.csv`, read
from the scanned figures; the electron density at 80 mA, which is not in the figure, is the 2e10
cm-3 of the text of Gordiets et al. 1996 (J. Phys. D 29, 1021). The digitized curves used for the
radiative states and the flow rate are in `docs/literatura/dados/levaton2020` and `levaton2021`,
with the scripts that extracted them.

The quantities compared are labelled as in the scripts: V1, the sustaining E/N; V2, the
vibrational temperature; V3, the ratio N2(B)/N2(C); V4, the electron density against the flow rate;
V5, the dissociation degree [N]/[N2].

## Chemistries

| Case | Chemistry |
|---|---|
| `n2_amonia` | A: the N2 part of the N2-H2 mechanism of Sun et al. 2022 (`casos/xu2025` without H) |
| `n2_tese` | B: the N2 part of the N2-O2 mechanism of ZDPlasKin (`casos/ar_n2_o2` without O and Ar) |
| `n2_exemplo` | C: the N2 example of ZDPlasKin (the mechanism of `casos/n2`) |
| `n2_amonia_v45`, `n2_tese_v45` | A and B with the vibrational ladder of N2(X) up to v = 45 |

Their mechanisms belong to other authors and are not distributed. Rebuild first the cases
`xu2025`, `ar_n2_o2` and `n2` as in `casos/README.md`; then `python3 ferramentas/casos_coluna.py`
writes the `kinet.inp` and `bolsigdb.dat` of the five cases (the same files, byte for byte, as
those of the article; `dbd1d/testes/test_casos_coluna.py`), and `python3 ferramentas/caso.py all
casos/NAME` builds each one. The other files of these cases are distributed. The cases used for the
tests of mechanisms (`n2_amonia_superelastica`, `n2_amonia_etapas`, `n2_amonia_acoplado`) are
built by the scripts that use them.

## Results and how to reproduce them

`scripts/coluna_n2_todos.sh [PROCESSES]` runs everything below in order, then writes the figures
of the article with `python3 scripts/figuras_coluna_n2.py artigo_coluna_n2/figuras`. The coupled
runs dominate the cost: 52 chains of four iterations of 20 to 40 minutes each.

| File | Script | Content |
|---|---|---|
| `etapa4/resultados.csv` | `coluna_n2.py` | Tv (V2) and dissociation (V5) at the measured E/N, the three chemistries truncated at v = 8, with the envelope of the error bars |
| `etapa5/en.csv` | `coluna_n2_en.py` | sustaining E/N (V1), with the associative ionization multiplied by 0.2 to 5 and T_g at the edges of its bar |
| `etapa6/v3.csv` | `coluna_n2_v3.py` | N2(B)/N2(C) at the end of the column (V3) |
| `falta/vazao.csv` | `coluna_n2_vazao.py` | electron density and E/N against the flow rate at 60 mA and 230 Pa (V4) |
| `falta/parede.csv` | `coluna_n2_parede.py` | sensitivity to the wall probabilities |
| `falta/convergencia.csv` | `coluna_n2_convergencia.py` | numerical convergence of the 0D model |
| `falta/superelastica.csv` | `coluna_n2_superelastica.py` | superelastic collisions with a prescribed vibrational distribution |
| `falta/etapas.csv` | `coluna_n2_etapas.py` | stepwise ionization from N2(A), N2(B) and N2(a') |
| `falta/sondas.csv` | `coluna_n2_sondas.py` | the electron balance imposed at t and at t/2 (the probes) |
| `falta/escada_v45.csv` | `coluna_n2_lote.py` | the ladder up to v = 45 with each set of wall probabilities |
| `falta/acoplado/*.json` | `coluna_n2_acoplado.py` | the electron-vibration coupling, one file per iteration |

The names of the coupled runs are the condition, `_vib_gordiets` for the vibrational wall
deactivation of Gordiets et al. (7e-3; without it, the 4.5e-4 of Levaton et al.) and the step of the
decomposition: `_alvo` (N2(v) as electron targets with the cross sections of N2 as they are),
`_deslocado` (with the thresholds lowered by E_v), none for the full coupling (with the superelastic
collisions), `_meio` and `_limiar01` for the two tests of sensitivity (half the shift, minimum
shifted threshold of 0.1 eV).

The folder names (`etapa4` to `etapa6`, steps of the original plan, and `falta`, the remaining
analyses) are kept as they were when the results were produced.
