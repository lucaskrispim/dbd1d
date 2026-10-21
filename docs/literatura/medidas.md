# Registro das medidas para validar o simulador 1D

Releitura completa dos artigos de `docs/artigos/` em 20/10/2026. Para cada artigo: as
condições do experimento (entrada do simulador), cada grandeza medida ou calculada que dá para
comparar, de onde ela saiu, o arquivo com os números e a grandeza do simulador que corresponde a
ela. Os números foram extraídos das figuras por `ferramentas/digitaliza.py` (vetorial quando o
PDF permite, raster nos demais); cada artigo tem um script em `docs/literatura/dados/` que refaz a
extração e registra a calibração dos eixos.

## Como o simulador produz cada grandeza

| Grandeza | Saída do simulador | Conta |
|---|---|---|
| Carga no eletrodo Q(t) | `superficie.dat`: V aplicada, φ esq | Q = ε₀ ε_d (V − φ_esq) A / Δ (lei de Gauss na barreira) |
| Corrente externa I(t) | idem | I = dQ/dt; o dt de gravação precisa resolver o pico (Stankov: largura de ~1 µs) |
| Tensão no gap U_gap(t) | `superficie.dat`: φ esq, φ dir | U_gap = φ_esq − φ_dir |
| Potência média | idem | P = (1/T) ∮ V dQ num período (área da figura de Lissajous) |
| Energia por pulso | idem | E = ∫ V dQ no pulso, de antes da subida até o fim dos refletidos |
| Tensão de ruptura | idem | U_gap no instante do pico principal de corrente |
| Tensão de ignição | varredura de amplitude | menor amplitude com corrente de descarga periódica em todos os períodos |
| n_e médio | `perfil.dat` | média no gap, ponderada pelo tamanho das células |
| Densidade de N, H, N₂(A) | `perfil.dat` | média no gap e valor no centro (posição da sonda não informada) |
| T_g | `perfil.dat`, com `--aquecimento` | média no gap |
| Parâmetros de enxame | `tabela_bolsig.dat` do caso | v_d = μE, D/μ, μN e energia média contra E/N |

## Argônio

### Stankov et al. 2020, PSST 29 125009 (prioridade 1)

Condições: DBD plano-paralela simétrica, eletrodos de cobre com quartzo de Δ = 1 mm em cada um,
gap d = 3 mm, área A = 7,26 cm², volume 2,178 cm³. Senoide de 24 kHz (T = 41,67 µs); pares de
pressão e amplitude medidos: 100 mbar e 1,8 kV, 300 mbar e 2,2 kV, 500 mbar e 2,7 kV,
650 mbar e 3,4 kV. Argônio em fluxo (Venturi), T_g = 300 K (aquecimento desprezível
segundo a ref. [25] do artigo), descarga difusa em todos os casos. Sinais medidos como média de 512
aquisições com amostragem de 1 ns. Modelo deles: fluido 1D com energia dos elétrons, 22 espécies,
γ = 0,02 e ε_r = 4,2 de melhor ajuste, 1500 elementos no gás (o menor de 0,15 µm), dez períodos
até o regime periódico. Na figura 4 a tensão aplicada é U_a(t) = −U₀ sen(2π t/T) com t = 0 no
início de um período do regime periódico.

| Grandeza | Condição | Fonte | Arquivo em `dados/stankov2020/` | Observação |
|---|---|---|---|---|
| Corrente I(t), um período | 100 mbar, 1,8 kV | fig. 3, medida | `fig3_corrente_medida_100mbar.csv`, `fig3_pico_medido_100mbar.csv`, `fig3_corcova_medida_100mbar.csv` | pico de 11,8 mA a 18,53 µs (1,6 mA/cm²); vetorial, passo de 4 ns no detalhe do pico |
| Corrente I(t), dez períodos | 650 mbar, 3,4 kV | fig. 6, medida | `fig6_corrente_medida_650mbar.csv` | o PDF guarda a curva com passo de 0,525 µs e 0,7 mA: serve para o instante dos picos, não para a amplitude |
| Potência média contra p | os quatro pares | figs. 7 e 12, medida | `fig7_potencia.csv` | 0,559, 0,935, 1,535 e 2,474 W; as duas figuras concordam em 0,003 W |
| Potência calculada por eles, por γ e por ε_r | idem | figs. 7 e 12 | `fig7_potencia.csv`, `fig12_potencia_eps.csv` | referência modelo a modelo |
| U_gap(t), U_a(t), n̄_e(t) calculados | 100 mbar, 1,8 kV, cinco γ | fig. 4 | `fig4_*.csv` | referência modelo a modelo |
| Tensão de ruptura, I_max, n_e,max | 100 mbar, 1,8 kV | tabelas 2 e 3 | `tabela2_gama.csv`, `tabela3_eps.csv` | n_e,max bate com o máximo da fig. 4(b) nos cinco γ (até 1 %); a tensão de ruptura bate com o máximo de U_gap da fig. 4(a) só para γ de 0,01 a 0,04 (a 0,005, 680 contra 649 V), então a tabela não define ruptura como o máximo de U_gap: comparar U_gap no pico de corrente; o I_max da tabela (129 a 164 mA) é dez vezes o pico do modelo na fig. 3 (~16 mA, coerente com os ~2 mA/cm² do texto), provável erro do artigo, não usar |

Também no texto: o pico principal dura cerca de 1 µs; a 300 e 500 mbar o modelo deles fica
caótico, e a 650 mbar repete a cada cinco períodos com γ = 0,02 e 0,04; a ionização por
quimioionização de metaestáveis (Ar[1s₅] e Ar[1s₃]) domina no intervalo entre descargas.

### Loffhagen et al. 2020, PCPP, doi 10.1007/s11090-020-10121-y (prioridade 1)

Condições: plano-paralela, eletrodos de malha de aço sobre vidro borofloat de Δ = 2 mm com
ε_r = 4,6, gap d = 1 mm, área de 8 cm² (8 cm × 1 cm), 1 atm, 300 K, argônio 99,9999 %, fluxo de
6 slm, senoide de 86,2 kHz (T = 11,6 µs).

| Grandeza | Condição | Fonte | Arquivo em `dados/loffhagen2020/` | Observação |
|---|---|---|---|---|
| Tensão de ignição U_i (amplitude) | Ar puro | fig. 4, medida | `fig4_ignicao_medida.csv` (x = 0) | 3,02 kV, barra de 2,87 a 3,17 kV |
| U_i contra ppm de TMS | 5 a 200 ppm | fig. 4, medida | idem | só serve com a química do TMS |
| U_i calculada por eles | 0 a 200 ppm | fig. 4 | `fig4_ignicao_modelo_*.csv` | no Ar puro, 3,65 kV: o modelo deles fica 0,6 kV acima da medida |

A ignição é definida no artigo como a amplitude em que a descarga acende e cobre toda a área do
eletrodo. No Ar puro, 60 % dos elétrons vêm de ionização direta e 40 % de quimioionização entre
átomos excitados (texto da fig. 5).

### Crispim et al. 2024, Phys. Scr. 99 065521

Condições do texto: 1 atm, 1 kV a 10 MHz, gap de 8 mm, 100 nós, 1 µs, n_e(0) = 1 cm⁻³, cinco
espécies. O código de 2024 usa gap de 0,8 mm (`Lx = 8.0d-4`); ver `extracao.md`.

| Grandeza | Fonte | Arquivo em `dados/crispim2024/` | Observação |
|---|---|---|---|
| v_d contra E/N | fig. 7: Pack e Phelps 1961, Ellis 1976, Kucukarpaci e Lucas 1981, Nielsen 1936 (medidas) | `fig7_*.csv` | de 6e-4 a 690 Td |
| D/μ contra E/N | fig. 8: Warren e Parker 1962 (77 K), Townsend e Bailey 1922 (77 K e 288 K), medidas | `fig8_*.csv` | |
| μN contra E/N | fig. 9: Kucukarpaci e Lucas 1981, Robertson 1977, Pack 1992, Errett 1951, medidas | `fig9_*.csv` | |
| Energia média contra E/N | fig. 10(b): Escada 2007 e Benhenni 2005, calculadas | `fig10b_*.csv` | |

Esses dados testam as seções de choque do `bolsigdb.dat` e o BOLSIG+, não o fluido.

## N₂ e N₂/H₂

### Yang et al. 2022, PSST 31 015017 (prioridade 1 para a amônia)

Condições: canal de quartzo de seção 16 mm × 8 mm, eletrodos de cobre de 60 mm × 12 mm por fora
das paredes de cima e de baixo, gap de 8 mm (o texto diz *electrode gap (8 mm compared to 10 mm)* e dá a seção do canal; que os
8 mm são entre as paredes internas é inferência). Espessura da parede não
informada; o mesmo laboratório usa 1,75 mm (Takashima 2012). 150 Torr, fluxo de 1 SLM, gás a
temperatura ambiente na entrada. Pulsos de ~10 ns e até 30 kV (FID), 100 kHz, rajadas de 5 a 150
pulsos, 10 Hz entre rajadas (100 ms de pós-descarga antes de cada rajada). Gases: N₂ puro e N₂ com
1 a 33 % de H₂ (fig. 16: 0, 1, 5, 9, 17 e 33 %; a energia por pulso foi medida de 1 a 20 %). TALIF com o feixe ao longo do canal e coleta pela parede lateral (posição no gap
não informada); TDLAS em passagem única ao longo de 6 cm. A onda de ionização fica a ~1 mm da
parede do cátodo (pulso curto); o plasma é difuso até o pulso 100 e começa a filamentar depois.
O modelo deles é quase 0D com a potência medida como entrada.

| Grandeza | Condição | Fonte | Arquivo em `dados/yang2022/` | Observação |
|---|---|---|---|---|
| V(t), I(t) do pulso 50 | N₂ | fig. 2(a) | `fig2_tensao_pulso50.csv`, `fig2_corrente_pulso50.csv` | entrada do simulador (forma do pulso, com os refletidos até 200 ns); pico de 25,2 kV no CSV (mediana por coluna, que corta o pico) e 26,2 kV pela borda da linha |
| P(t) e energia acumulada do pulso 50 | N₂ | fig. 2(b) | `fig2_potencia_pulso50.csv`, `fig2_energia_pulso50.csv` | 3,3 mJ no fim; a integral da potência digitalizada dá 3,7 mJ, 11 % a mais, vale a curva de energia |
| Energia por pulso na rajada | N₂ | fig. 3 | `fig3_energia_total.csv`, `fig3_energia_incidente.csv` | de 3,48 a 3,08 mJ (total) e de 2,2 a 2,0 mJ (incidente) |
| N₂(A, v=0 e 1) no tempo | N₂, 150 pulsos | fig. 7 | `fig7_n2a_v0_N2.csv`, `fig7_n2a_v1_N2.csv` | envoltória de cima e de baixo; picos de 3,6e13 e 8,8e13 cm⁻³ perto de 50 µs |
| T_rot-trans na rajada | N₂ | fig. 8(b) | `fig8_temperatura_N2.csv` | de 327 a 407 K em 1,5 ms; incerteza de 10 K |
| N acumulado e N por pulso | N₂ | fig. 9 | `fig9_N_acumulado_N2.csv`, `fig9_N_por_pulso_N2.csv` | 4,0e15 cm⁻³ no pulso 150 |
| N₂(A, v=0 e 1) no tempo | N₂ + 1 % H₂ | fig. 11 | `fig11_*.csv` | picos menores pela desativação por H |
| N acumulado, N por pulso e H acumulado | N₂ + 1 % H₂ | fig. 14 | `fig14_*.csv` | N de 1,0e15 e H de 5,9e15 cm⁻³ no fim da rajada |
| N e H contra fração de H₂ | 50 pulsos, 0 a 33 % | fig. 16 | `fig16_N_contra_H2.csv`, `fig16_H_contra_H2.csv` | N cai de 2,3e15 (0 %) a 2,2e14 (33 %); H sobe de 1,9e15 (1 %) a 3,8e15 |
| Decaimento no pós-descarga | N₂ e N₂ + 1 % H₂ | figs. 10 e 15 | não extraído | escalas de ms; depende de perdas na parede e fluxo |

Consistência entre figuras: N depois de 50 pulsos em N₂ dá 2,32e15 (fig. 9) e 2,35e15 (fig. 16);
H depois de 50 pulsos com 1 % de H₂ dá 1,83e15 (fig. 14(b)) e 1,92e15 (fig. 16).

### Takashima et al. 2012, AIAA 2012-3093

Condições: ar seco, canal de quartzo de 220 mm × 22 mm × 10 mm (altura, o gap), paredes de
1,75 mm, eletrodos de cobre de 14 mm × 65 mm por fora, de 20 a 340 Torr (CPT) ou até 500 Torr
(FID). Gerador CPT: ~50 ns, 25 kV, 20 e 40 kHz. Gerador FID: 5 a 10 ns, 30 kV, 10 kHz. Lâmpada UV
para ruptura no primeiro pulso.

| Grandeza | Fonte | Arquivo em `dados/takashima2012/` | Observação |
|---|---|---|---|
| Energia por pulso contra p, CPT | fig. 8, direita | `fig8_energia_CPT_40kHz.csv`, `fig8_energia_CPT_20kHz.csv` | cresce até ~2,1 mJ em ~200 Torr e cai a zero em 340 Torr |
| Energia por pulso contra p, FID | fig. 10, direita | `fig10_energia_FID_incidente.csv`, `fig10_energia_FID_incidente_e_refletidos.csv` | incidente até 2,9 mJ; com refletidos até 4,2 mJ |
| Campo aplicado no tempo, 60 Torr | figs. 7 e 9 | `fig7_campo_aplicado_CPT.csv`, `fig9_campo_aplicado_FID.csv` | entrada do simulador: tensão = campo × (d + 2 l/ε); CPT de −21 a +17 kV/cm, FID até 25 kV/cm |
| Tensão de ruptura | texto | 13 kV medida a 60 Torr (CPT) | |

## Artigos que não entram na comparação quantitativa

| Artigo | Motivo |
|---|---|
| Redzikultsava et al. 2026, PSST 35 035019 | Ar a 1,3 Torr, gap de 10 mm, assimétrica: medem corrente (fig. 4) e temperatura de excitação por OES (fig. 6), que não é T_e do modelo; candidato a comparação qualitativa da forma da corrente |
| Mrkvičková et al. 2016, PSST 25 055015 | H absoluto por TALIF em DBD coplanar de superfície em Ar/H₂ a 1 atm (acima de 1e22 m⁻³); geometria fora do 1D entre placas |
| Minesi et al. 2021, PSST 30 035008 | ZDPlasKin 0D com campo fixo em faísca de ar, n_e até 1e19 cm⁻³; fora do regime de DBD |
| Zhao et al. 2024, ICOPS (Ar-NH₃) | só o resumo de uma página, sem números |
