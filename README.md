# The Physical Structure and Evolution of Transition Region Explosive Events Observed with ESIS

Roy T. Smart, Charles C. Kankelborg, and Jacob D. Parker

The slides are at <https://roytsmart.github.io/solar-research-2026/>.

To preview them locally, run `python scripts/serve.py` and open
<http://localhost:8000/docs/>. Unlike `python -m http.server`, it answers the
range requests a browser needs to seek in the movies.

## Abstract

The EUV Snapshot Imaging Spectrograph (ESIS) was launched on a NASA sounding rocket in September 2019. ESIS is a computed tomography imaging spectrograph (CTIS), which records a spectrum in every pixel across a wide field of view simultaneously, with high spatial, spectral, and temporal resolution. This snapshot capability lets ESIS capture the full spatial and spectral structure of rapidly evolving features that slit-scanning instruments such as IRIS can only sample sequentially. During its 5-minute flight, ESIS observed ~20 transition region explosive events (EEs), compact brightenings with supersonic wing enhancements, in O V 630 Å. Leveraging simultaneous imaging and spectroscopy, this presentation will characterize the physical structure and dynamical evolution of these EEs, and place them into the context of other EE observations.

## Reproducing the figures

The figures were made in August 2026, and most of the packages they depend on
have moved on since. They were not all made in the same environment either:
the simulation results were computed on 2026-08-08, and the Level-4 figures
from the flight were made on 2026-08-10 to 2026-08-12, after `esis` had been
switched to the branch holding the Level-4 product. Reproducing them therefore
takes two virtual environments, both held to the third-party versions of the
time by `constraints.txt`.

The versions were read off the git reflog of each package, which records what
was checked out when, and the times each cached result was computed, which
`joblib` stores beside it. Several were development commits rather than
releases, which is why they are installed from GitHub by commit.

**The simulation figures**, and everything else not made from the Level-4
product, use the environment `pyproject.toml` describes:

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -c constraints.txt -e . jupyter
.venv/Scripts/python -m ipykernel install --user --name solar-research-2026
```

The last line registers the environment as a Jupyter kernel, so that the
notebooks can be run against the pinned versions rather than against
whatever the default kernel has installed.

**The Level-4 figures** use `requirements-level-4.txt`, which differs in its
`esis` and its `sdo`:

```bash
python -m venv .venv-level-4
.venv-level-4/Scripts/python -m pip install -r requirements-level-4.txt
.venv-level-4/Scripts/python -m pip install --no-deps -e .
```

The Level-4 product, `esis.data.Level_4`, was never released, and only exists
on the `wip/level4-tempest-tooling` branch of `esis`. The `sdo` those figures
were made with was a commit partway along the branch which added `sdo.hmi`.
The branch was squash-merged as sun-data/solar-dynamics-observatory#20 and
deleted, and GitHub still serves the commit because the pull request's head
descends from it. The Level-4 FITS cubes
themselves (7.5 GB per line) are read from `~/.spd2026/data`, falling back to
the published copy on the `Z:` share, which is about an hour slower. They are
only ever read, never written, so this repository and the talk share them.

The package is still called `spd2026`, as it was in the talk, but it keeps its
cache in `~/.solar-research-2026/cache` rather than in the talk's
`~/.spd2026/cache`. `joblib` files each result under the name of the function
which computed it and throws the result away when the code of that function
changes, so two repositories sharing one cache would throw away each other's
results. `esis` has a cache of its own with the same weakness. The `esis` of
the Level-4 environment lets `ESIS_CACHE_DIR` move it, but the `esis` of the
simulation environment predates that variable and always uses
`~/.esis/cache`, which it shares with every other `esis` on the machine.

Each figure is a function in `spd2026.figures` which writes into `figures/`
by default, for example:

```python
import spd2026
spd2026.figures.mart_scene()
spd2026.figures.level_4()
```

Several of the figures in `figures/` were made with arguments other than the
defaults. `scripts/regenerate_figures.py` holds the call behind every one of
them, and makes the ones belonging to the environment it is run in:

```bash
.venv/Scripts/python scripts/regenerate_figures.py simulation
.venv-level-4/Scripts/python scripts/regenerate_figures.py level-4
```

Run from empty caches, with `ESIS_CACHE_DIR` pointing at an empty directory,
every IRIS, Level-1 and Level-4 figure it has been checked against came out
identical to the one the talk showed: the movies byte for byte, and the SVGs
but for the random IDs matplotlib gives the parts of every SVG it writes.
The despiked IRIS raster and the synthetic scene are bit-identical to the
talk's too. The figures made from the simulated images cannot be: `optika`
2.1.0 samples the field and the pupil at random, and adds read noise, without
a seed, so every linearization and every image is a new draw. Rerun, the effective area of
each channel moves by a percent or two and the noise is different, and the
figures made from them show a different realization of the same simulation.
The one the talk showed is kept in the talk's cache, `~/.spd2026/cache`.

Regenerating takes about an hour and a half, and a lot of memory. The
backprojection behind the two blink figures asks for 123 GB at once, and each
Level-4 figure holds an 18 GB cube, so neither should share the machine with
anything large, or with each other.
