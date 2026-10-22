# Cases

A case is a folder with the chemistry and the electron tables of one gas mixture:

| File | Content | Produced by |
|---|---|---|
| `kinet.inp` | reaction mechanism in ZDPlasKin format | the user |
| `bolsigdb.dat` | electron cross sections in BOLSIG+ format (LXCat) | the user (download from lxcat.net) |
| `composicao.txt` | background gas for BOLSIG+: `SPECIES FRACTION` or `SPECIES traco` per line | the user |
| `gas.json` | background gas, heat capacity, thermal conductivity, seed ion (optional enthalpies) | the user, or `caso.py enthalpies` |
| `transporte.json` | ion mobilities, neutral diffusion, wall loss | `caso.py transport` / `caso.py estimate` |
| `quimica.json` | the chemistry read from `zdplaskin_m.F90`, plus the BOLSIG+ thresholds | `caso.py export` |
| `tabela_*.dat` | electron coefficients from BOLSIG+ versus E/N, mean energy and gas temperature | `caso.py tables` |

Generating a case needs ZDPlasKin and BOLSIG+ (free for non-commercial use, not redistributable,
download from www.zdplaskin.laplace.univ-tlse.fr) and gfortran:

```
export ZDPLASKIN_DIR=/path/to/ZDPlasKin_2.0a_Linux     # preprocessor, dvode_f90_m.F90, bolsig_x86_64.so
python3 ferramentas/caso.py all casos/NAME
```

The actions are `zdplaskin`, `tables`, `transport`, `estimate`, `enthalpies`, `heat`, `reduce`,
`export` and `all`; the Portuguese names of earlier versions are still accepted.

Running a case needs only `quimica.json`, the tables, `gas.json` and `transporte.json`:
`python3 dbd1d/simula.py OUTPUT --case casos/NAME ...`.

## Cases used in the article that are not distributed

Their mechanisms belong to other authors and their cross sections come from LXCat, which does not
allow redistribution. Their folders hold only the files written by the dbd1d authors (`gas.json`,
`transporte.json`, `composicao.txt` and, where they exist, `monta.py`, `gera_dados.py`,
`t_nos.txt`, `en_max.txt`). To rebuild a case, add the files below and run `caso.py all`.

| Case | `kinet.inp` | `bolsigdb.dat` | Composition |
|---|---|---|---|
| `n2` | N2 DBD test case of the ZDPlasKin website | N2 set of the same example | `N2 1` |
| `amonia_sun2022`, `yang2022_N2`, `yang2022_1pcH2`, `xu2025` | N2-H2 mechanism of the supplementary data of Sun et al., Plasma Sources Sci. Technol. 31 (2022) 094009 | the supplementary BOLSIG+ file of the same article | `N2 0.5 / H2 0.5`, `N2 1`, `N2 0.99 / H2 0.01`, `N2 0.25 / H2 0.75` |
| `ar_n2` | the argon mechanism of this repository and the N2 mechanism of the case `n2`, joined with three cross reactions by `casos/ar_n2/monta.py` (a test mixture of the tool) | the argon and N2 sets of the two cases | `Ar 0.9 / N2 0.1` |
| `ar_n2_o2` | N2-O2 mechanism of the ZDPlasKin website (version 1.03, with the processes that have no cross sections in the database commented out), joined to the argon mechanism of this repository with six cross reactions by `casos/ar_n2_o2/monta.py` | the N2 and O2 sets used with that mechanism plus argon | `N2 0.78 / O2 0.21 / Ar 0.01` |

Variants built from the cases above by editing `kinet.inp` and running
`caso.py zdplaskin` and `caso.py export` (the tables do not change, since only heavy-particle
coefficients change):

| Case | Change |
|---|---|
| `yang2022_N2_pool50`, `yang2022_1pcH2_pool50` | `N2(A3) + N2(A3)`: 1.6e-10 to `N2 + N2(B3)`, 0.8e-10 to `N2 + N2(C3)`, 2.4e-10 to `N2 + N + N` cm3/s (50% dissociation, total kept at 4.8e-10) |
| `yang2022_1pcH2_h05`, `yang2022_1pcH2_pool50_h05` | `N2(a'1) + H2 => N2 + H + H` split into 1.3e-11 to `N2 + H + H` and 1.3e-11 to `N2 + H2` cm3/s (50% H yield), the second also with the pooling change |
| `amonia_sun2022_hongW98` | `N + H2(v) => NH + H` in the corrected form of Hong et al., J. Phys. D 51 (2018) 109501 (W98): 4e-10 (Tg/300)^0.5 exp(-(16600 - 0.3 Ev)/Tg), with Ev from the BOLSIG+ thresholds |
| `yang2022_N2_calor` | `yang2022_N2` with the heat block in `gas.json` (rotational excitation of N2 as fast heating, enthalpies by `caso.py enthalpies` plus six ions given in the file); needs `caso.py heat` and `caso.py tables` |

The case `argonio_jovanovic` is complete and runs as is: the argon mechanism at the repository root with
`Ar* + Ar* => Ar^+ + Ar + e` at 1.2e-9 cm3/s and `Ar* + Ar + Ar` at 8.33e-33 cm6/s, from
Jovanović et al., Plasma Sources Sci. Technol. (2023), doi 10.1088/1361-6595/acd2fc.

The cases `argonio_jovanovic_ci` and `argonio_jovanovic_tc` change only one of the two coefficients
(chemi-ionization and three-body quenching) and are complete too.

The argon case at the repository root is complete and runs as is: its mechanism (`kinet.inp`)
takes the heavy-particle coefficients of the argon example distributed with ZDPlasKin, with two
reactions added by the authors, and its tables come from the SIGLO argon cross sections (LXCat).
