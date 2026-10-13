# 4D_MRI_GNL_correction

[![DOI](https://zenodo.org/badge/1414122614.svg)](https://doi.org/10.5281/zenodo.23348757)

Open-source correction of gradient-nonlinearity (GNL) velocity errors in phase-contrast and 4D flow MRI.

Velocity encoding assumes spatially linear gradient fields. Real gradient coils deviate from linearity with distance from the isocentre, so the measured velocity vector is scaled and rotated by a position-dependent 3×3 matrix Λ(r). This toolbox implements the generalised velocity reconstruction of Markl et al. (2003), v_corr = Λ(r)⁻ᵀ · v_meas, voxel by voxel, using the manufacturer's spherical-harmonic description of the gradient coil (the Siemens `.grad` coefficient file). It was developed and validated in the master thesis *Open-Source Preprocessing for Gradient Nonlinearity Correction in 4D Flow MRI* (Jaina Jaison, Otto von Guericke University Magdeburg, 2026).

**Release used for the thesis:** v1.0.0, archived on Zenodo: https://doi.org/10.5281/zenodo.23348758 (this version); https://doi.org/10.5281/zenodo.23348757 always resolves to the latest version.

## What it does

One run of the toolbox performs, in order:

1. **Coil characterisation** – parses the `.grad` file (Schmidt / L4 normalisation), evaluates the spherical-harmonic field, builds Λ(r) on the acquisition grid and reports strength factors, encoding tilt, condition number and the spatially varying VENC.
2. **Synthetic verification** – distorts a known velocity field (Mode 1: steady Poiseuille tube; Mode 2: pulsatile helical tube) with the same Λ, adds noise at a chosen SNR, corrects it and reports RMSE / MAE / 95th-percentile error before and after correction, an SNR sweep and six built-in self-tests (including a model-mismatch test with perturbed coefficients).
3. **Correction of acquired data** – loads a 4D flow DICOM series (three velocity components, optional magnitude), verifies the geometry (patient position, table position, obliquity, slice spacing, distortion-correction state), maps the voxel positions to the gradient-coil frame, applies the correction to every voxel and cardiac phase, and writes the results.

Outputs per run (written to the output folder): five figures (`fig01` distortion matrix, `fig02` synthetic error maps, `fig03` spatially varying VENC, `fig04` synthetic summary metrics, `fig05` correction of the acquired data), a complete run record `run_config.json`, `velocity_field.vtk` for ParaView and `dicom_corrected.npz` with the measured and corrected fields.

## Repository contents

```
4D_MRI_GNL_correction.py   the toolbox (single Python module)
run_thesis.sh              the exact commands used for the thesis datasets
CITATION.cff               citation metadata
LICENSE                    MIT
```

The repository contains the code only. The outputs of the thesis runs (the run records `run_config.json`, the figures, the ParaView state files and the VTK/NPZ velocity fields) are not kept here because of their size; the run records and ParaView states are available from the author on request. The DICOM datasets (phantom and volunteer data) are not distributed. The manufacturer's coefficient files (`.grad`) are not included either: they are the vendor's property and are obtained from the scanner or from the vendor on request; the toolbox reads them as they come.

## Requirements

Python 3.10 or newer with `numpy`, `scipy`, `matplotlib` and `pydicom`; `tkinter` (part of the standard Python installation) for the graphical file picker.

```
pip install numpy scipy matplotlib pydicom
```

## Usage

**Graphical picker** – start the script without arguments; a dialog asks for the coefficient file, the DICOM folder, the output folder, the dataset preset and the velocity-axis mapping:

```
python 4D_MRI_GNL_correction.py
```

**Command line** – all options of the dialog are available as arguments (`python 4D_MRI_GNL_correction.py --help`). The commands used for the thesis datasets:

```
# 3 T pipe-flow phantom (steady flow, VENC 7 cm/s)
python 4D_MRI_GNL_correction.py --grad coeff_skyra.grad --dicom <pipe series folder> \
    --preset pipe_flow --venc 0.07 --sim-mode 1 --out <output folder>

# 7 T idealised aneurysm phantom (VENC 150 cm/s)
python 4D_MRI_GNL_correction.py --grad coeff_sc72.grad --dicom <phantom series folder> \
    --preset ideal_phantom --direction-permute yxz --direction-signs=1,1,-1 --out <output folder>

# 7 T circle of Willis, in vivo (VENC 90 cm/s)
python 4D_MRI_GNL_correction.py --grad coeff_sc72.grad --dicom <in vivo folder> \
    --preset in_vivo_head --direction-permute yxz --direction-signs=-1,-1,1 --out <output folder>

# coil characterisation and synthetic verification only (no DICOM)
python 4D_MRI_GNL_correction.py --grad coeff_IMPULSE.grad --preset ideal_phantom --out <output folder>
```

Key options:

| Option | Meaning |
|---|---|
| `--preset {pipe_flow, ideal_phantom, in_vivo_head}` | masking and figure settings for the type of dataset |
| `--venc M_S` | VENC of the acquired data in m/s, if it is not found in the DICOM header |
| `--direction-permute`, `--direction-signs` | order and sign of the three stored velocity components relative to the patient axes; needed for exports that lack the DICOM velocity-direction tags (see thesis, Section 3.6.1). The mapping is recorded in the run record. |
| `--snr-db`, `--sim-venc`, `--sim-nt`, `--sim-mode`, `--flow-pattern` | settings of the synthetic verification |
| `--no-npz` | skip the large `.npz` export |

Every run writes `run_config.json`, a complete record of the settings, geometry checks, self-test results and statistics of that run; the run records of the thesis runs reproduce every number reported in the thesis.

## Method in brief

- Λ(r) is obtained by central differences of the modelled field of each coil (Equation 3.4 of the thesis) and is evaluated only inside the calibration sphere r ≤ R₀ of the coefficient file.
- The inverse is applied through a singular-value gate (σ_min > 10⁻⁴) and an adaptive Tikhonov regularisation that is inactive for well-conditioned voxels (κ ≤ 2) and reaches λ = 0.05 at κ ≥ 10.
- The correction is multiplicative; additive phase errors (eddy currents, concomitant fields) are not addressed and must be removed separately.

## Citation

If you use this toolbox, please cite the thesis and the archived release (see `CITATION.cff`):

> Jaison J. *Open-Source Preprocessing for Gradient Nonlinearity Correction in 4D Flow MRI.* Master thesis, Otto von Guericke University Magdeburg, 2026. Toolbox v1.0.0: https://github.com/jainajaison13-source/4D_MRI_GNL_correction, DOI 10.5281/zenodo.23348758.

Underlying method: Markl M, Bammer R, Alley MT, et al. Generalized reconstruction of phase contrast MRI: analysis and correction of the effect of gradient field distortions. *Magn Reson Med* 2003;50:791–801.

## Licence

MIT License – see `LICENSE`.
