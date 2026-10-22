# Validation experiments

Each experiment reproduces the conditions of an article in the record (`docs/literatura/medidas.md`)
to compare with the quantities extracted in `docs/literatura/dados/`. The runs come from the
scripts `remoto/validacao_*.sh`; the comparisons, from `scripts/compara_enxame.py`, `compara_stankov.py`,
`detalha_stankov.py`, `compara_loffhagen.py`, `compara_takashima.py` and `compara_yang.py`.

Order of work in each experiment: cost of one period (or pulse), mesh and tolerance convergence at
one condition, production at the other conditions with the chosen mesh, comparison.

## V1. Stankov et al. 2020: argon, current and power

| Parameter | Value | Option of `simula.py` |
|---|---|---|
| Gap, barriers | 3 mm; 1 mm quartz, ε_r = 4.2, on both electrodes | `--gap 3e-3 --thickness 1e-3 --eps-dielectric 4.2` |
| Voltage | U_a = −U₀ sin(2π t/T), 24 kHz | `--waveform sine --v-peak -U0 --period 4.1666666666666665e-05` |
| Pressure and amplitude | 100 mbar and 1.8 kV; 300 and 2.2; 500 and 2.7; 650 and 3.4 | `--pressure 1e4 --v-peak -1800` etc. |
| Gas | argon at 300 K, without heating (the authors neglect it) | argon case at the repository root, `--t-gas 300` |
| Surface | γ = 0.02; secondary electrons with 2 eV | `--gamma 0.02 --eps-sec 2` |
| Electrons | energy equation | `--energy` |
| Initial state | quasi-neutral, 1e12 m⁻³ of each charged species | `--neutral --ne0 1e6` |
| Duration | ten periods (417 µs), comparison in the tenth | `--t-end` = 10 T |
| Area for the current | 7.26 cm² | in the comparison |

Model differences that remain: five-species chemistry (they have 22, with 1s and 2p levels,
excimers and chemi-ionization by metastables); electron reflection at the walls (they use
r_e = 0.7, available as `--electron-reflection 0.7`); ion mobilities of `transporte.json`.

## V2. Loffhagen et al. 2020: argon, ignition voltage at 1 atm

| Parameter | Value | Option |
|---|---|---|
| Gap, barriers | 1 mm; 2 mm borofloat, ε_r = 4.6, on both | `--gap 1e-3 --thickness 2e-3 --eps-dielectric 4.6` |
| Voltage | U_a = U₀ sin(2π f t), 86.2 kHz | `--waveform sine --period 1.1600928074245939e-05` |
| Scan | U₀ from 2.0 to 4.0 kV in steps of 0.25 | `--v-peak` |
| Gas | argon at 1 atm and 300 K | argon case at the repository root |
| Surface | γ = 0.02 (not given in the article; the value of the same group in Stankov 2020), 2 eV | `--gamma 0.02 --eps-sec 2` |
| Duration | ten periods (116 µs) | |
| Ignition criterion | in the last period, current peak above twice the peak of the capacitive current without plasma | in the comparison |

## V3. Yang et al. 2022: N₂ and N₂ with 1% H₂, burst of nanosecond pulses

| Parameter | Value | Option |
|---|---|---|
| Gap, barriers | 8 mm; 1.75 mm quartz (thickness of the same laboratory in Takashima 2012; not given in the article), ε_r = 3.8 | `--gap 8e-3 --thickness 1.75e-3 --eps-dielectric 3.8` |
| Voltage | measured pulse 50, repeated at 100 kHz | `--waveform table --voltage-table validacao/yang2022/pulso50.txt --period 1e-5` |
| Pressure | 150 Torr | `--pressure 19998` |
| Gas | pure N₂ and N₂:H₂ 99:1, ammonia mechanism of Sun 2022 | `--case casos/yang2022_N2`, `--case casos/yang2022_1pcH2` |
| Temperature | 300 K, with heating (the article measures T) | `--t-gas 300 --t-wall 300 --heating` |
| Surface | γ = 0.05, secondary electrons with 5 eV (program default; not given in the article) | `--gamma 0.05 --eps-sec 5` |
| Electrons | energy equation | `--energy` |
| Initial state | residue of the previous burst (100 ms before) unknown | `--neutral --ne0 1e6`, with sensitivity |
| Duration | 150 pulses (1.5 ms) | `--t-end 1.5e-3` |
| Area for the energy | 60 mm × 12 mm = 7.2 cm² | in the comparison |

Model differences that remain: the dissociation branch of N₂(A) + N₂(A) pooling is 6% in the
mechanism (3e-11 of 4.8e-10 cm³/s) and the article concludes 50% (the variant cases `*_pool50`
test that); N₂(A) without vibrational resolution (compare the total with the measured v = 0 plus
v = 1, which are part of it); no flow (residence time of ~0.1 s, much longer than the 1.5 ms burst).

## V4. Swarm parameters in argon (Crispim 2024, figures 7 to 10)

No simulation: `tabela_bolsig.dat` at the repository root (BOLSIG+ with `bolsigdb.dat`) against the data of the
figures. v_d = μN · (E/N); D/μ = DN/μN; mean energy = 3/2 k T_e.

## V5. Takashima et al. 2012: dry air, energy per pulse versus pressure

| Parameter | Value | Option |
|---|---|---|
| Gap, barriers | 10 mm; 1.75 mm quartz, ε_r = 3.8 (not given) | `--gap 1e-2 --thickness 1.75e-3 --eps-dielectric 3.8` |
| Voltage | applied field measured at 60 Torr times d + 2Δ/ε_r (1.092 cm), CPT generator at 40 kHz | `--waveform table --voltage-table validacao/takashima2012/pulso_cpt.txt --period 2.5e-5` |
| Gas | N₂ 78%, O₂ 21%, Ar 1% (N₂-O₂ mechanism of ZDPlasKin with argon) | `--case casos/ar_n2_o2` |
| Pressures | 20, 40, 60, 100, 200 and 300 Torr | `--pressure` |
| Surface | γ = 0.05, secondary electrons with 5 eV (program default; not given in the article) | `--gamma 0.05 --eps-sec 5` |
| Duration | five pulses; energy of the fifth by a fine restart (`scripts/retoma_fina.py`) | |
| Area | 14 mm × 65 mm = 9.1 cm² | in the comparison |

The pulse shape measured at 60 Torr is used at all pressures (the article does not show the others);
above ~100 Torr the measured discharge stops being uniform, and the 1D model does not represent that.

## Townsend breakdown (verification)

`validacao/townsend.py` compares the breakdown voltage of the discrete model, from the growth rate
of the ion inventory in short runs at fixed voltage, with a continuous linear reference that uses
the same wall conditions as the code (options `--meshes`, `--gamma`, `--electron-reflection`, `--output`,
`--reference-only`).

## Short validation

`validacao/curta.py` runs reduced versions of V1, V2, V4 and the Townsend verification in a few
minutes and checks them against reference values of the same short runs and against the measurements;
the test `prototipo/testes/test_validacao_curta.py` runs its fast items in every test run and the
complete set when the slow tests are selected.
