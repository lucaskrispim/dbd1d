# Register of the measurements used to validate the 1D simulator

Complete rereading of the articles in `docs/artigos/` on 20/10/2026. For each article: the
conditions of the experiment (input of the simulator), each measured or computed quantity that can
be compared, where it came from, the file with the numbers and the corresponding quantity of the
simulator. The numbers were extracted from the figures by `ferramentas/digitaliza.py` (vector when
the PDF allows, raster otherwise); each article has a script in `docs/literatura/dados/` that redoes
the extraction and records the calibration of the axes.

## How the simulator produces each quantity

| Quantity | Output of the simulator | Computation |
|---|---|---|
| Electrode charge Q(t) | `superficie.dat`: applied V (`V aplicada`), φ left (`phi esq`) | Q = ε₀ ε_d (V − φ_left) A / Δ (Gauss's law in the barrier) |
| External current I(t) | same | I = dQ/dt; the output dt must resolve the peak (Stankov: width of ~1 µs) |
| Gap voltage U_gap(t) | `superficie.dat`: φ left, φ right | U_gap = φ_left − φ_right |
| Mean power | same | P = (1/T) ∮ V dQ over one period (area of the Lissajous figure) |
| Energy per pulse | same | E = ∫ V dQ over the pulse, from before the rise to the end of the reflections |
| Breakdown voltage | same | U_gap at the instant of the main current peak |
| Ignition voltage | amplitude scan | lowest amplitude with a periodic discharge current in every period |
| Mean n_e | `perfil.dat` | gap average, weighted by the cell size |
| Density of N, H, N₂(A) | `perfil.dat` | gap average and value at the centre (probe position not given) |
| T_g | `perfil.dat`, with `--heating` | gap average |
| Swarm parameters | `tabela_bolsig.dat` of the case | v_d = μE, D/μ, μN and mean energy versus E/N |

## Argon

### Stankov et al. 2020, PSST 29 125009 (priority 1)

Conditions: symmetric plane-parallel DBD, copper electrodes each covered by quartz of Δ = 1 mm, gap
d = 3 mm, area A = 7.26 cm², volume 2.178 cm³. Sine of 24 kHz (T = 41.67 µs); measured pairs of
pressure and amplitude: 100 mbar and 1.8 kV, 300 mbar and 2.2 kV, 500 mbar and 2.7 kV, 650 mbar and
3.4 kV. Flowing argon (Venturi), T_g = 300 K (heating negligible according to ref. [25] of the
article), diffuse discharge in all cases. Signals measured as the mean of 512 acquisitions sampled
at 1 ns. Their model: 1D fluid with electron energy, 22 species, γ = 0.02 and ε_r = 4.2 as best
fit, 1500 elements in the gas (the smallest of 0.15 µm), ten periods to the periodic regime. In
figure 4 the applied voltage is U_a(t) = −U₀ sin(2π t/T) with t = 0 at the start of a period of
the periodic regime.

| Quantity | Condition | Source | File in `dados/stankov2020/` | Remark |
|---|---|---|---|---|
| Current I(t), one period | 100 mbar, 1.8 kV | fig. 3, measured | `fig3_corrente_medida_100mbar.csv`, `fig3_pico_medido_100mbar.csv`, `fig3_corcova_medida_100mbar.csv` | peak of 11.8 mA at 18.53 µs (1.6 mA/cm²); vector, step of 4 ns in the detail of the peak |
| Current I(t), ten periods | 650 mbar, 3.4 kV | fig. 6, measured | `fig6_corrente_medida_650mbar.csv` | the PDF stores the curve with a step of 0.525 µs and 0.7 mA: good for the instant of the peaks, not for the amplitude |
| Mean power versus p | the four pairs | figs. 7 and 12, measured | `fig7_potencia.csv` | 0.559, 0.935, 1.535 and 2.474 W; the two figures agree within 0.003 W |
| Power computed by them, per γ and per ε_r | same | figs. 7 and 12 | `fig7_potencia.csv`, `fig12_potencia_eps.csv` | model to model reference |
| Computed U_gap(t), U_a(t), n̄_e(t) | 100 mbar, 1.8 kV, five γ | fig. 4 | `fig4_*.csv` | model to model reference |
| Breakdown voltage, I_max, n_e,max | 100 mbar, 1.8 kV | tables 2 and 3 | `tabela2_gama.csv`, `tabela3_eps.csv` | n_e,max agrees with the maximum of fig. 4(b) for the five γ (within 1%); the breakdown voltage agrees with the maximum of U_gap in fig. 4(a) only for γ from 0.01 to 0.04 (at 0.005, 680 against 649 V), so the table does not define breakdown as the maximum of U_gap: compare U_gap at the current peak; the I_max of the table (129 to 164 mA) is ten times the model peak of fig. 3 (~16 mA, consistent with the ~2 mA/cm² of the text), probably an error of the article, not used |

Also in the text: the main peak lasts about 1 µs; at 300 and 500 mbar their model becomes chaotic,
and at 650 mbar it repeats every five periods with γ = 0.02 and 0.04; chemi-ionisation of
metastables (Ar[1s₅] and Ar[1s₃]) dominates ionisation between discharges.

### Loffhagen et al. 2020, PCPP, doi 10.1007/s11090-020-10121-y (priority 1)

Conditions: plane-parallel, steel mesh electrodes on borofloat glass of Δ = 2 mm with ε_r = 4.6,
gap d = 1 mm, area of 8 cm² (8 cm × 1 cm), 1 atm, 300 K, argon 99.9999%, flow of 6 slm, sine of
86.2 kHz (T = 11.6 µs).

| Quantity | Condition | Source | File in `dados/loffhagen2020/` | Remark |
|---|---|---|---|---|
| Ignition voltage U_i (amplitude) | pure Ar | fig. 4, measured | `fig4_ignicao_medida.csv` (x = 0) | 3.02 kV, bar from 2.87 to 3.17 kV |
| U_i versus ppm of TMS | 5 to 200 ppm | fig. 4, measured | same | only usable with the TMS chemistry |
| U_i computed by them | 0 to 200 ppm | fig. 4 | `fig4_ignicao_modelo_*.csv` | in pure Ar, 3.65 kV: their model lies 0.6 kV above the measurement |

The article defines ignition as the amplitude at which the discharge ignites and covers the whole
electrode area. In pure Ar, 60% of the electrons come from direct ionisation and 40% from
chemi-ionisation between excited atoms (text of fig. 5).

### Crispim et al. 2024, Phys. Scr. 99 065521

Conditions of the text: 1 atm, 1 kV at 10 MHz, gap of 8 mm, 100 nodes, 1 µs, n_e(0) = 1 cm⁻³, five
species. The 2024 code uses a gap of 0.8 mm (`Lx = 8.0d-4`); see `extracao.md`.

| Quantity | Source | File in `dados/crispim2024/` | Remark |
|---|---|---|---|
| v_d versus E/N | fig. 7: Pack and Phelps 1961, Ellis 1976, Kucukarpaci and Lucas 1981, Nielsen 1936 (measured) | `fig7_*.csv` | from 6e-4 to 690 Td |
| D/μ versus E/N | fig. 8: Warren and Parker 1962 (77 K), Townsend and Bailey 1922 (77 K and 288 K), measured | `fig8_*.csv` | |
| μN versus E/N | fig. 9: Kucukarpaci and Lucas 1981, Robertson 1977, Pack 1992, Errett 1951, measured | `fig9_*.csv` | |
| Mean energy versus E/N | fig. 10(b): Escada 2007 and Benhenni 2005, computed | `fig10b_*.csv` | |

These data test the cross sections of `bolsigdb.dat` and BOLSIG+, not the fluid model.

## N₂ and N₂/H₂

### Yang et al. 2022, PSST 31 015017 (priority 1 for ammonia)

Conditions: quartz channel of cross section 16 mm × 8 mm, copper electrodes of 60 mm × 12 mm outside
the top and bottom walls, gap of 8 mm (the text says *electrode gap (8 mm compared to 10 mm)* and
gives the channel cross section; that the 8 mm lie between the inner walls is an inference). Wall
thickness not given; the same laboratory uses 1.75 mm (Takashima 2012). 150 Torr, flow of 1 SLM, gas
at room temperature at the inlet. Pulses of ~10 ns and up to 30 kV (FID), 100 kHz, bursts of 5 to
150 pulses, 10 Hz between bursts (100 ms of afterglow before each burst). Gases: pure N₂ and N₂ with
1 to 33% H₂ (fig. 16: 0, 1, 5, 9, 17 and 33%; the energy per pulse was measured from 1 to 20%). TALIF
with the beam along the channel and collection through the side wall (position in the gap not
given); single-pass TDLAS along 6 cm. The ionisation wave stays ~1 mm from the cathode wall (short
pulse); the plasma is diffuse up to pulse 100 and starts to filament afterwards. Their model is
quasi-0D with the measured power as input.

| Quantity | Condition | Source | File in `dados/yang2022/` | Remark |
|---|---|---|---|---|
| V(t), I(t) of pulse 50 | N₂ | fig. 2(a) | `fig2_tensao_pulso50.csv`, `fig2_corrente_pulso50.csv` | input of the simulator (pulse shape, with the reflections up to 200 ns); peak of 25.2 kV in the CSV (median per column, which cuts the peak) and 26.2 kV from the edge of the line |
| P(t) and cumulative energy of pulse 50 | N₂ | fig. 2(b) | `fig2_potencia_pulso50.csv`, `fig2_energia_pulso50.csv` | 3.3 mJ at the end; the integral of the digitised power gives 3.7 mJ, 11% more; the energy curve is used |
| Energy per pulse in the burst | N₂ | fig. 3 | `fig3_energia_total.csv`, `fig3_energia_incidente.csv` | from 3.48 to 3.08 mJ (total) and from 2.2 to 2.0 mJ (incident) |
| N₂(A, v=0 and 1) in time | N₂, 150 pulses | fig. 7 | `fig7_n2a_v0_N2.csv`, `fig7_n2a_v1_N2.csv` | upper and lower envelope; peaks of 3.6e13 and 8.8e13 cm⁻³ near 50 µs |
| T_rot-trans in the burst | N₂ | fig. 8(b) | `fig8_temperatura_N2.csv` | from 327 to 407 K in 1.5 ms; uncertainty of 10 K |
| Accumulated N and N per pulse | N₂ | fig. 9 | `fig9_N_acumulado_N2.csv`, `fig9_N_por_pulso_N2.csv` | 4.0e15 cm⁻³ at pulse 150 |
| N₂(A, v=0 and 1) in time | N₂ + 1% H₂ | fig. 11 | `fig11_*.csv` | smaller peaks due to quenching by H |
| Accumulated N, N per pulse and accumulated H | N₂ + 1% H₂ | fig. 14 | `fig14_*.csv` | N of 1.0e15 and H of 5.9e15 cm⁻³ at the end of the burst |
| N and H versus H₂ fraction | 50 pulses, 0 to 33% | fig. 16 | `fig16_N_contra_H2.csv`, `fig16_H_contra_H2.csv` | N falls from 2.3e15 (0%) to 2.2e14 (33%); H rises from 1.9e15 (1%) to 3.8e15 |
| Afterglow decay | N₂ and N₂ + 1% H₂ | figs. 10 and 15 | not extracted | ms scales; depends on wall losses and flow |

Consistency between figures: N after 50 pulses in N₂ gives 2.32e15 (fig. 9) and 2.35e15 (fig. 16);
H after 50 pulses with 1% H₂ gives 1.83e15 (fig. 14(b)) and 1.92e15 (fig. 16).

### Takashima et al. 2012, AIAA 2012-3093

Conditions: dry air, quartz channel of 220 mm × 22 mm × 10 mm (height, the gap), walls of 1.75 mm,
copper electrodes of 14 mm × 65 mm outside, from 20 to 340 Torr (CPT) or up to 500 Torr (FID). CPT
generator: ~50 ns, 25 kV, 20 and 40 kHz. FID generator: 5 to 10 ns, 30 kV, 10 kHz. UV lamp for
breakdown in the first pulse.

| Quantity | Source | File in `dados/takashima2012/` | Remark |
|---|---|---|---|
| Energy per pulse versus p, CPT | fig. 8, right | `fig8_energia_CPT_40kHz.csv`, `fig8_energia_CPT_20kHz.csv` | grows up to ~2.1 mJ at ~200 Torr and falls to zero at 340 Torr |
| Energy per pulse versus p, FID | fig. 10, right | `fig10_energia_FID_incidente.csv`, `fig10_energia_FID_incidente_e_refletidos.csv` | incident up to 2.9 mJ; with reflections up to 4.2 mJ |
| Applied field in time, 60 Torr | figs. 7 and 9 | `fig7_campo_aplicado_CPT.csv`, `fig9_campo_aplicado_FID.csv` | input of the simulator: voltage = field × (d + 2 l/ε); CPT from −21 to +17 kV/cm, FID up to 25 kV/cm |
| Breakdown voltage | text | 13 kV measured at 60 Torr (CPT) | |

## Articles left out of the quantitative comparison

| Article | Reason |
|---|---|
| Redzikultsava et al. 2026, PSST 35 035019 | Ar at 1.3 Torr, gap of 10 mm, asymmetric: they measure current (fig. 4) and an excitation temperature by OES (fig. 6), which is not the T_e of the model; candidate for a qualitative comparison of the current shape |
| Mrkvičková et al. 2016, PSST 25 055015 | absolute H by TALIF in a coplanar surface DBD in Ar/H₂ at 1 atm (above 1e22 m⁻³); geometry outside the 1D parallel-plate setting |
| Minesi et al. 2021, PSST 30 035008 | 0D ZDPlasKin with fixed field in an air spark, n_e up to 1e19 cm⁻³; outside the DBD regime |
| Zhao et al. 2024, ICOPS (Ar-NH₃) | only a one-page abstract, without numbers |
