# Loops cannot see most registration error

Code and per-landmark results for the methods note *Loops cannot see most registration error: measuring the visibility ceiling of cycle consistency* (Russell Parrish, Parallax Metrology, draft, 20 November 2026). Read the note: [parallaxmetrology.com/loop-visibility](https://parallaxmetrology.com/loop-visibility/) or [`note/index.html`](note/index.html).

## The measurement

Consistency and transitivity checks (register A to B, B to C and A to C, and see whether the loop closes) are used as label-free quality checks on deformable image registration. True displacements are differences of positions, so they close around every loop. The part of a registration set that fails to close is therefore exactly the part of its *error* that fails to close. Splitting the error on the registration graph into its integrable (gradient) and cyclic parts, wherever ground truth exists, gives the share of error that any closure-based check can see. If errors were independent per registration, the invisible share on a complete graph of *n* images would be 2/*n*.

| Registrar, dataset | Landmarks | Invisible share | 95% interval (patients resampled) | After annotation-noise allowance | Independent-error baseline |
|---|---:|---:|---:|---:|---:|
| Diffeomorphic Demons, DIR-Lab 4DCT (10 patients, 6 phases, 15 pairs) | 750 | 93.1% | 90.3 to 96.2% | 91.2% | 33.3% |
| 32-voxel translation window, DIR-Lab | 477 | 79.1% | 73.3 to 95.8% | 77.2% | 33.3% |
| 32-voxel translation window, POPI (1 patient, 10 phases, 45 pairs) | 28 | 92.9% | 88.0 to 96.4% (landmarks resampled) | not identifiable | 20.0% |

The allowance is deliberately pessimistic: it reads DIR-Lab's published mean repeat-annotation distance as the error of a single annotation. On POPI, whose errors are small and which publishes no repeat-annotation figure, annotation noise cannot be excluded as the main invisible component. The visible part still names a landmark's worst registration above chance (18.4% against 6.7% on DIR-Lab), but no better than an NCC image residual (20.5%). A pre-registered attempt to read the error shared between the two registrars as annotation noise (triple collocation) failed its known-answer test against DIR-Lab's published observer errors; the result is reported in the note.

## Check it

```
pip install numpy
python reproduce.py      # recomputes every number in the note from derived/ (81 checks)
python check_hashes.py   # study files match the SHA-256 hashes recorded before each run
```

`reproduce.py` needs no images or landmark files. Point estimates are checked to 1e-6; bootstrap intervals are recomputed with the script's own resampling and must fall within 1.5 percentage points of those reported; rank correlations get 1e-3 (see the comment in the script).

## Layout

| Path | What it is |
|---|---|
| `note/index.html`, `note/labbook.html` | The note (public edition: Figure 4 shows trajectories on a 5 mm grid without the CT image) and the full lab book |
| `derived/` | Per-landmark results: integrable and cyclic error energy, mean edge error, noise allowance, worst-edge indicators. No coordinates, displacement vectors or image data |
| `results/` | Summary outputs of every analysis, as written by the study scripts |
| `instrument/` | Standalone copies of the two registrars (`window.py`, `demons.py`) and an independent Hodge split (`hodge_split.py`) |
| `verification/instrument_equivalence.json` | On DIR-Lab data, the standalone instrument reproduced the study's sealed outputs exactly: 282 + 60 windows (translations, refinements, residuals, guard decisions), 505 landmark graphs (Hodge energies to 1.5e-15 relative), and one Demons pair (75 displacements, 0.0 mm difference) |
| `study/` | The exact plans, freeze and seal records and scripts as run, in their original layout. The scripts import a private pilot codebase that is not published; the functions they call are reproduced in `instrument/` |

## What is not here, and how to get it

DIR-Lab 4DCT images and landmarks are distributed by the DIR-Lab group at Emory University through an access request form ([DIR-Lab 4DCT](https://med.emory.edu/departments/radiation-oncology/research-laboratories/deformable-image-registration/downloads-and-reference-data/4dct.html)); POPI is described at [open-vv/popi-model](https://github.com/open-vv/popi-model/blob/master/popi-model.md). Neither the images, the landmark coordinates nor the raw measurement files (which contain rounded landmark positions) are redistributed here. `study/repo_build/export_derived.py` shows exactly how `derived/` was produced from them.

## Provenance

The POPI analysis is exploratory (POPI had been examined in a preceding pilot). On DIR-Lab, the covariance test was pre-registered before any data were on the machine; the visibility measurements were declared and hashed before each run, and measurements were sealed before target landmarks were read. The Demons all-pairs run was interrupted by an out-of-memory crash after five cases; cases 6 to 10 were completed by a memory-lean script with identical parameters, declared and hashed before it ran (`study/dirlab/ADDENDUM_2b.json`). Analysis code was written with AI coding assistants (OpenAI Codex for the preceding pilot, Anthropic Claude for this study).

## Requirements

`reproduce.py`, `check_hashes.py`: Python 3.10+, numpy. `instrument/`: numpy, scipy, scikit-image, and SimpleITK 2.5.8 for Demons (see `requirements.txt`).

## Licence and citation

Code: MIT ([`LICENSE`](LICENSE)). Note, lab book, derived results and documentation: CC BY 4.0 ([`LICENSE-CONTENT`](LICENSE-CONTENT)). To cite, see [`CITATION.cff`](CITATION.cff).
