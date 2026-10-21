# dbd1d

One-dimensional fluid model of dielectric barrier discharges (DBD) built automatically from the
chemistry files of ZDPlasKin: a `kinet.inp` mechanism and a BOLSIG+ cross-section file give a
complete model (species, reactions, electron coefficients, transport) without editing code.

- drift-diffusion transport of every charged and neutral species, Poisson equation with the
  charge accumulated on the dielectric barriers, electron energy equation (local mean energy
  approximation) and gas heating;
- reaction rates read from the `zdplaskin_m.F90` that the ZDPlasKin preprocessor generates,
  reproducing the rates and source terms of ZDPlasKin itself to 1e-11 (including user variables,
  IF blocks, single precision literals and integer division as in Fortran);
- adaptive BDF integration with the electric potential in the state vector and an analytic sparse
  Jacobian;
- applied voltage: sine, step, trapezoidal pulse train or a tabulated measured pulse repeated
  periodically;
- optional electron reflection at the walls (`--electron-reflection`, Hagelaar et al. 2000).

Options, messages and comments are in English. Identifiers, file names and the keys of the data
files keep their original Portuguese names, and the earlier Portuguese option names (`--pressao`,
`--forma seno`, ...) are still accepted as aliases.

## Installation

Python 3.10 or newer:

```
pip install -r requirements.txt
OPENBLAS_NUM_THREADS=1 python3 -m pytest -q -m "not lento"
```

The digitizer of the validation data reads vector figures with `pdftocairo` (Debian and Ubuntu
package `poppler-utils`); without it, its four vector tests are skipped.

Tested with numpy 2.2 and 2.5, scipy 1.15 and 1.18. The integrator builds on the internal BDF
implementation of scipy (`scipy.integrate._ivp`), so scipy is bounded below 1.19 until a newer
version is tested. In this package 371 tests pass and 109 are
skipped, each with the reason: they generate cases (ZDPlasKin and BOLSIG+ needed) or use the
gas mixtures whose mechanisms and cross sections we cannot redistribute (`casos/LEIAME.md`). With
`ZDPLASKIN_DIR` set and those cases rebuilt, they run too.

## Quick start

```
exemplos/quickstart.sh
```

runs three periods of an argon DBD (Stankov et al. 2020, 100 mbar, 1.8 kV at 24 kHz) and plots the
discharge current against the measurement. Expected output (about 3 minutes): in the third period,
mean power 0.647 W (measured 0.559 W) and current peak of about 24 mA at 19.6 µs.

## Running

```
python3 prototipo/simula.py OUTPUT_DIR [--case casos/NAME] --gap 3e-3 --thickness 1e-3 \
    --eps-dielectric 4.2 --pressure 1e4 --waveform sine --period 4e-5 --v-peak 1800 \
    --energy --t-end 4e-4 --dt-output 5e-8
python3 prototipo/simula.py --help
```

`--t-end` must be a whole number of `--dt-output` intervals.

Output: `superficie.dat` (applied voltage, potentials and surface charge of both barriers) and
`perfil.dat` (profiles of every species, fields, electron energy and gas temperature), plus
`parametros.json` with the arguments, the sha256 of every input and the code version. A run can be
resumed from any saved instant with `--restart DIR T`, repeating the options of the original run
(the program refuses a different mesh, case or voltage form).

## Data

The argon tables were generated with the argon cross sections of the SIGLO database
(www.lxcat.net, L. C. Pitchford and J. P. Boeuf), which is not redistributed here. The measured
data in `docs/literatura/dados/` were digitized from the figures of the cited articles for comparison
and must be cited to them.

## New gas mixtures

See `casos/LEIAME.md`. Generating a case needs ZDPlasKin and BOLSIG+, which are free for
non-commercial use but cannot be redistributed: download them from
www.zdplaskin.laplace.univ-tlse.fr and set `ZDPLASKIN_DIR`. Running a case does not need them.

## Validation

`validacao/LEIAME.md` defines the comparisons of the article (Stankov et al. 2020, Loffhagen et
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
package and keep their own conditions of use.
