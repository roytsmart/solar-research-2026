# The Physical Structure and Evolution of Transition Region Explosive Events Observed with ESIS

Roy T. Smart, Charles C. Kankelborg, and Jacob D. Parker

## Abstract

The EUV Snapshot Imaging Spectrograph (ESIS) was launched on a NASA sounding rocket in September 2019. ESIS is a computed tomography imaging spectrograph (CTIS), which records a spectrum in every pixel across a wide field of view simultaneously, with high spatial, spectral, and temporal resolution. This snapshot capability lets ESIS capture the full spatial and spectral structure of rapidly evolving features that slit-scanning instruments such as IRIS can only sample sequentially. During its 5-minute flight, ESIS observed ~20 transition region explosive events (EEs), compact brightenings with supersonic wing enhancements, in O V 630 Å. Leveraging simultaneous imaging and spectroscopy, this presentation will characterize the physical structure and dynamical evolution of these EEs, and place them into the context of other EE observations.

## Reproducing the figures

The figures were made in August 2026 against versions of `optika`, `named-arrays`,
`regridding`, `ctis` and `esis` which have since moved on, and the
intermediate results cached in `~/.spd2026/cache` were computed with those
versions. `pyproject.toml` therefore pins every home-grown dependency to the
version used at the time, and the package should be installed into its own
virtual environment rather than alongside a current checkout of the stack:

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -e .
```

The Level-4 flight product, `esis.data.Level_4`, was never released. It comes
from the `wip/level4-tempest-tooling` branch of `esis`, which the pin installs
straight from GitHub at the commit the talk used. The Level-4 FITS cubes
themselves (7.5 GB per line) are read from `~/.spd2026/data`, falling back to
the published copy on the `Z:` share, which is about an hour slower.

Each figure is a function in `spd2026.figures` which writes into `figures/`
by default, for example:

```python
import spd2026
spd2026.figures.mart_scene()
spd2026.figures.level_4()
```
