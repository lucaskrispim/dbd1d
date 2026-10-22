# dbd1d

One-dimensional fluid model of dielectric barrier discharges (DBD) built automatically from the
chemistry files of ZDPlasKin: a `kinet.inp` mechanism and a BOLSIG+ cross-section file give a
complete model (species, reactions, electron coefficients, transport) without editing code.

- drift-diffusion transport of every charged and neutral species, Poisson equation with the
  charge accumulated on the dielectric barriers, electron energy equation (local mean energy
  approximation) and gas heating;
- reaction rates read from the `zdplaskin_m.F90` that the ZDPlasKin preprocessor generates,
  reproducing the rates and source terms of ZDPlasKin itself to a relative accuracy of 1e-11 (including user variables,
  IF blocks, single-precision literals and integer division as in Fortran);
- adaptive BDF integration of a differential-algebraic system with the electric potential in the
  state vector and a structured sparse Jacobian (coloured finite differences for the local
  terms, exact rows for the Poisson equation);
- applied voltage: sine, step, trapezoidal pulse train or a tabulated measured pulse repeated
  periodically;
- optional electron reflection at the walls (`--electron-reflection`, Hagelaar et al. 2000).

Options, messages and comments are in English. Identifiers, file names and the keys of the data
files keep their original Portuguese names, and the earlier Portuguese option names (`--pressao`,
`--forma seno`, ...) are still accepted as aliases. Up to version 1.2.2 the folder `dbd1d/` was
called `prototipo/`.

## Layout of the repository

| Folder | Content |
|---|---|
| `dbd1d/` | the simulator: `simula.py` (command line), `modelo.py` (model), `jacobiano.py` and `bdf_dae.py` (integration), `quimica_zdp.py` (chemistry), `tabela.py` (electron tables); tests in `dbd1d/testes/` |
| `ferramentas/` | tools: `caso.py` builds a case from ZDPlasKin files, with the Fortran programs that generate the electron tables; the digitizer of published figures and an electron swarm Monte Carlo |
| `casos/` | cases (gas mixtures); the argon case is at the repository root; see `casos/README.md` |
| `biblioteca/` | library of transport data per species and background gas |
| `validacao/` | validation and verification: `curta.py` (short validation), `townsend.py`, the input pulses of the pulsed cases; see `validacao/README.md` |
| `scripts/` | comparison and analysis scripts (`compara_*.py`, `resumo.py`, `fluxos.py`, `retoma_fina.py`, ...) |
| `remoto/` | scripts that start the validation runs (`validacao_*.sh`) |
| `docs/literatura/` | record of the measurements and the data digitized from the articles, with the extraction scripts |
| `artigo/` | the figure script of the article and `artigo/README.md`, the command of every run |
| `exemplos/` | `quickstart.sh` |
| `referencias/` | reference outputs of the regression test |

Folder and file names are in Portuguese (*ferramentas*: tools, *casos*: cases, *biblioteca*: library,
*validacao*: validation, *remoto*: remote, *referencias*: references); the options, messages,
comments and documentation are in English.

## Installation

You need Python 3.10 or newer and git (or the zip of the release on Zenodo,
https://doi.org/10.5281/zenodo.23535841). The program runs from the folder of the repository; the
installation only creates a Python virtual environment with its dependencies, so that their versions
do not interfere with other Python projects on the machine.

Linux and macOS (on Debian and Ubuntu, first install the packages that the virtual environment
needs: `sudo apt install python3-venv git`):

```
git clone https://github.com/lucaskrispim/dbd1d
cd dbd1d
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 validacao/curta.py
```

Windows (PowerShell; the program should run there, but it was not tested on Windows):

```
git clone https://github.com/lucaskrispim/dbd1d
cd dbd1d
py -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python validacao/curta.py
```

The last command checks the installation in a few minutes (see Short validation below) and ends
with `N of N checks passed`. Afterwards, activate the environment again in each new terminal
(`source .venv/bin/activate`, or `.venv\Scripts\activate` on Windows) before running the program.
To run the automated tests:

```
OPENBLAS_NUM_THREADS=1 python3 -m pytest -q -m "not lento"
```

Running the cases is pure Python and was tested on Linux (Ubuntu 22.04 and 24.04); it should run
wherever numpy and scipy run. Generating a new case (`ferramentas/caso.py`) needs ZDPlasKin, whose BOLSIG+ library is
distributed for Linux x86-64, and a Fortran compiler (gfortran).

The digitizer of the validation data reads vector figures with `pdftocairo` (Debian and Ubuntu
package `poppler-utils`); without it, its four vector tests are skipped.

Tested with numpy 2.2 and 2.5, scipy 1.15 and 1.18. The integrator builds on the internal BDF
implementation of scipy (`scipy.integrate._ivp`), so scipy is bounded below 1.19 until a newer
version is tested. With the command above, 451 tests pass, 59 are skipped and 14 slow ones (marked
`lento`) are deselected; without `-m "not lento"`, 72 are skipped. Each skipped test states, before
running, the file it needs: ZDPlasKin and BOLSIG+ (to generate cases), the gas mixtures whose
mechanisms and cross sections we cannot redistribute (`casos/README.md`), or the old Fortran code of
the research repository with its references. With `ZDPLASKIN_DIR` set and those cases rebuilt, the
first two groups run too.

## Short validation

```
python3 validacao/curta.py
```

runs reduced versions of the comparisons of the article in a few minutes (about 4 on a laptop) and
checks them: the argon DBD of Stankov et al. 2020 for 3 periods with 50 cells, the ignition of argon
at 1 atm below and above the measured band of Loffhagen et al. 2020, the Townsend breakdown against
its continuous reference and the electron swarm data of argon. The full runs of the article, which
take hours to days, are listed with their commands in `artigo/README.md`.

## Quick start

```
exemplos/quickstart.sh
```

runs three periods of an argon DBD (Stankov et al. 2020, 100 mbar, 1.8 kV at 24 kHz) and plots the
discharge current against the measurement. Expected output (about 3 minutes): in the third period,
mean power 0.647 W (measured 0.559 W) and current peak of about 24 mA at 19.6 µs.

## Running

```
python3 dbd1d/simula.py OUTPUT_DIR [--case casos/NAME] --gap 3e-3 --thickness 1e-3 \
    --eps-dielectric 4.2 --pressure 1e4 --waveform sine --period 4e-5 --v-peak 1800 \
    --energy --t-end 4e-4 --dt-output 5e-8
python3 dbd1d/simula.py --help
```

`--t-end` must be a whole number of `--dt-output` intervals.

Output: `superficie.dat` (applied voltage, potentials and surface charge of both barriers) and
`perfil.dat` (profiles of every species, fields, electron energy and gas temperature), plus
`parametros.json` with the arguments, the sha256 of every input and the code version. A run can be
resumed from any saved instant with `--restart DIR T`, repeating the options of the original run.
The restart refuses a different mesh, voltage waveform or initial gas temperature and refuses to drop
the energy, heating or free-gas variables; other options may change, for sensitivity studies from
a saved state, and the program lists every option that differs from the original run.

## Data

The argon tables were generated with the argon cross sections of the SIGLO database
(www.lxcat.net, L. C. Pitchford and J. P. Boeuf), which is not redistributed here. The measured
data in `docs/literatura/dados/` were digitized from the figures of the cited articles for comparison
and must be credited to those articles.

## New gas mixtures

See `casos/README.md`. Generating a case needs ZDPlasKin and BOLSIG+, which are free for
non-commercial use but cannot be redistributed: download them from
www.zdplaskin.laplace.univ-tlse.fr and set `ZDPLASKIN_DIR`. Running a case does not need them.

## Validation

`validacao/README.md` defines the comparisons of the article (Stankov et al. 2020, Loffhagen et
al. 2020, Yang et al. 2022, Takashima et al. 2012, electron swarm data); the measured data
extracted from the figures of these articles are in `docs/literatura/dados/`, each with the script
that extracts them. The runs are started by `remoto/validacao_*.sh` and compared by
`scripts/compara_*.py`; `scripts/retoma_fina.py` restarts chosen pulses of a run with fine output.

`validacao/townsend.py` verifies the coupled model against the Townsend breakdown of a gap between
bare metal electrodes: it computes the breakdown voltage of the continuous linear problem with the
wall conditions of the code and finds the breakdown of the simulation from the growth rate of the
ion inventory (a few seconds per voltage).

## License

GPL-3.0-or-later (`LICENSE`). ZDPlasKin, BOLSIG+ and the LXCat cross sections are not part of this
package and keep their own conditions of use. `ferramentas/gera_tabela_energia.f90` and
`ferramentas/sonda_zdplaskin.f90` are compiled by `ferramentas/caso.py` on the user's machine
against the user's copy of ZDPlasKin and BOLSIG+; no combined program is distributed. The argon
mechanism at the repository root takes its heavy-particle coefficients from the argon example of
ZDPlasKin, and its tables were generated from the SIGLO cross sections (LXCat).
