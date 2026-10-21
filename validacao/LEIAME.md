# Experimentos de validação

Cada experimento reproduz as condições de um artigo do registro (`docs/literatura/medidas.md`)
para comparar com as grandezas extraídas em `docs/literatura/dados/`. As execuções saem dos
roteiros `remoto/validacao_*.sh`; as comparações, de `scripts/compara_enxame.py`, `compara_stankov.py`,
`detalha_stankov.py`, `compara_loffhagen.py`, `compara_takashima.py` e `compara_yang.py`.

Ordem de trabalho em cada experimento: custo de um período (ou pulso), convergência em malha e
em tolerância numa condição, produção nas demais com a malha escolhida, comparação.

## V1. Stankov et al. 2020: argônio, corrente e potência

| Parâmetro | Valor | Opção do `simula.py` |
|---|---|---|
| Gap, barreiras | 3 mm; quartzo de 1 mm, ε_r = 4,2, nos dois eletrodos | `--gap 3e-3 --espessura 1e-3 --eps-dieletrico 4.2` |
| Tensão | U_a = −U₀ sen(2π t/T), 24 kHz | `--forma seno --v-pico -U0 --periodo 4.1666666666666665e-05` |
| Pressão e amplitude | 100 mbar e 1,8 kV; 300 e 2,2; 500 e 2,7; 650 e 3,4 | `--pressao 1e4 --v-pico -1800` etc. |
| Gás | argônio a 300 K, sem aquecimento (os autores o desprezam) | caso da raiz, `--t-gas 300` |
| Superfície | γ = 0,02; elétrons secundários com 2 eV | `--gama 0.02 --eps-sec 2` |
| Elétrons | equação de energia | `--energia` |
| Estado inicial | quase neutro, 1e12 m⁻³ de cada espécie carregada | `--neutro --ne0 1e6` |
| Duração | dez períodos (417 µs), comparação no décimo | `--t-end` = 10 T |
| Área para a corrente | 7,26 cm² | na comparação |

Diferenças de modelo que ficam: química de cinco espécies (eles têm 22, com níveis 1s e 2p,
excímeros e quimioionização por metaestáveis); reflexão de partículas nas paredes (eles usam
r_e = 0,7); mobilidades iônicas do `transporte.json`.

## V2. Loffhagen et al. 2020: argônio, tensão de ignição a 1 atm

| Parâmetro | Valor | Opção |
|---|---|---|
| Gap, barreiras | 1 mm; borofloat de 2 mm, ε_r = 4,6, nos dois | `--gap 1e-3 --espessura 2e-3 --eps-dieletrico 4.6` |
| Tensão | U_a = U₀ sen(2π f t), 86,2 kHz | `--forma seno --periodo 1.1600928074245939e-05` |
| Varredura | U₀ de 2,0 a 4,0 kV de 0,25 em 0,25 | `--v-pico` |
| Gás | argônio a 1 atm e 300 K | caso da raiz |
| Superfície | γ = 0,02 (não informado no artigo; o valor do mesmo grupo em Stankov 2020), 2 eV | `--gama 0.02 --eps-sec 2` |
| Duração | dez períodos (116 µs) | |
| Critério de ignição | no último período, pico de corrente acima de duas vezes o pico da corrente capacitiva sem plasma | na comparação |

## V3. Yang et al. 2022: N₂ e N₂ com 1 % de H₂, rajada de pulsos de nanossegundos

| Parâmetro | Valor | Opção |
|---|---|---|
| Gap, barreiras | 8 mm; quartzo de 1,75 mm (espessura do mesmo laboratório em Takashima 2012; não informada no artigo), ε_r = 3,8 | `--gap 8e-3 --espessura 1.75e-3 --eps-dieletrico 3.8` |
| Tensão | pulso 50 medido, repetido a 100 kHz | `--forma tabela --tensao-tabela validacao/yang2022/pulso50.txt --periodo 1e-5` |
| Pressão | 150 Torr | `--pressao 19998` |
| Gás | N₂ puro e N₂:H₂ 99:1, mecanismo da amônia de Sun 2022 | `--caso casos/yang2022_N2`, `--caso casos/yang2022_1pcH2` |
| Temperatura | 300 K, com aquecimento (o artigo mede T) | `--t-gas 300 --t-parede 300 --aquecimento` |
| Elétrons | equação de energia | `--energia` |
| Estado inicial | resíduo da rajada anterior (100 ms antes) desconhecido | `--neutro --ne0 1e6`, com sensibilidade |
| Duração | 150 pulsos (1,5 ms) | `--t-end 1.5e-3` |
| Área para a energia | 60 mm × 12 mm = 7,2 cm² | na comparação |

Diferenças de modelo que ficam: o ramo de dissociação do agrupamento N₂(A) + N₂(A) é de 6 % no
mecanismo (3e-11 de 4,8e-10 cm³/s) e o artigo conclui 50 %; N₂(A) sem resolução vibracional
(comparar o total com v = 0 mais v = 1 medidos, que são parte dele); sem fluxo (residência de
~0,1 s, muito maior que a rajada de 1,5 ms).

## V4. Parâmetros de enxame em argônio (Crispim 2024, figuras 7 a 10)

Sem simulação: `tabela_bolsig.dat` da raiz (BOLSIG+ com o `bolsigdb.dat`) contra os dados das
figuras. v_d = μN · (E/N); D/μ = DN/μN; energia média = 3/2 k T_e.

## V5. Takashima et al. 2012: ar seco, energia por pulso contra a pressão

| Parâmetro | Valor | Opção |
|---|---|---|
| Gap, barreiras | 10 mm; quartzo de 1,75 mm, ε_r = 3,8 (não informado) | `--gap 1e-2 --espessura 1.75e-3 --eps-dieletrico 3.8` |
| Tensão | campo aplicado medido a 60 Torr vezes d + 2 l/ε (1,092 cm), gerador CPT a 40 kHz | `--forma tabela --tensao-tabela validacao/takashima2012/pulso_cpt.txt --periodo 2.5e-5` |
| Gás | N₂ 78 %, O₂ 21 %, Ar 1 % (mecanismo N₂-O₂ do ZDPlasKin com argônio) | `--caso casos/ar_n2_o2` |
| Pressões | 20, 40, 60, 100, 200 e 300 Torr | `--pressao` |
| Duração | cinco pulsos; energia do quinto por retomada fina | |
| Área | 14 mm × 65 mm = 9,1 cm² | na comparação |

A forma do pulso medida a 60 Torr vale para todas as pressões (o artigo não mostra as outras);
acima de ~100 Torr a descarga medida deixa de ser uniforme, e o 1D não representa isso.
