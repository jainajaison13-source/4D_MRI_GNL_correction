# 4D_MRI_GNL_correction

[![DOI](https://zenodo.org/badge/1414122614.svg)](https://doi.org/10.5281/zenodo.23348757)

Open-source correction of gradient-nonlinearity (GNL) velocity errors in phase-contrast and 4D flow MRI.

Velocity encoding assumes spatially linear gradient fields. Real gradient coils deviate from linearity with distance from the isocentre, so the measured velocity vector is scaled and rotated by a position-dependent 3×3 matrix Λ(r). This toolbox implements the generalised velocity reconstruction of Markl et al. (2003), v_corr = Λ(r)⁻ᵀ · v_meas, voxel by voxel, using the manufacturer's spherical-harmonic description of the gradient coil (the Siemens `.grad` coefficient file). It was developed and validated in the master thesis *Open-Source Preprocessing for Gradient Nonlinearity Correction in 4D Flow MRI* (Jaina Jaison, Otto von Guericke University Magdeburg, 2026).

**Release used for the thesis:** v1.0.1 (this release), archived on Zenodo; https://doi.org/10.5281/zenodo.23348757 always resolves to the latest version. v1.0.0 (https://doi.org/10.5281/zenodo.23348758) is the previous release; v1.0.1 adds the block-wise frame-layout option, the analysis scripts of the thesis figures and the text box of the summary figure (see `CHANGELOG.md`).

## What it does

One run of the toolbox performs, in order:

1. **Coil characterisation** – parses the `.grad` file (Schmidt quasi-normalisation, the file's lnorm = 4), evaluates the spherical-harmonic field, builds Λ(r) on the acquisition grid and reports strength factors, encoding tilt, condition number and the spatially varying VENC.
2. **Synthetic verification** – distorts a known velocity field (Mode 1: steady Poiseuille tube; Mode 2: pulsatile helical tube) with the same Λ, adds noise at a chosen SNR, corrects it and reports RMSE / MAE / 95th-percentile error before and after correction, an SNR sweep and six built-in self-tests (including a model-mismatch test with perturbed coefficients).
3. **Correction of acquired data** – loads a 4D flow DICOM series (three velocity components, optional magnitude), verifies the geometry (patient position, table position, obliquity, slice spacing, distortion-correction state), maps the voxel positions to the gradient-coil frame, applies the correction to every voxel and cardiac phase, and writes the results.

Outputs per run (written to the output folder): five figures (`fig01` distortion matrix, `fig02` synthetic error maps, `fig03` spatially varying VENC, `fig04` synthetic summary metrics, `fig05` correction of the acquired data), a complete run record `run_config.json`, `velocity_field.vtk` and `isocentre.vtk` for ParaView, and `dicom_corrected.npz` with the measured and corrected fields.

## Repository contents

```
4D_MRI_GNL_correction.py   the toolbox (single Python module)
run_thesis.sh              the exact commands used for the thesis datasets
Analysis/                  scripts that make the thesis figures and tables from the toolbox outputs
CHANGELOG.md               changes between releases
CITATION.cff               citation metadata
LICENSE                    MIT
```

The `Analysis/` scripts read only what the toolbox writes (`run_config.json`, `velocity_field.vtk`) and recompute nothing of the correction:

| Script | Thesis item |
|---|---|
| `pipe_profiles_figure.py` | Fig. 4.15, relative and plane-mean correction along the pipe |
| `pipe_raw_flow_figure.py` | Fig. 4.16, flow rate per cross-section before and after the correction |
| `cow_orientation_figure.py` | Fig. 4.20, orientation of the correction in vivo; Tab. 4.10 (dominant flow direction) |
| `cow_vessel_stats.py` | vessel-set numbers of Sections 4.5.1–4.5.3, Tab. 4.8 (vessel column) and Tab. 4.9 |
| `dicom_inventory.py`, `dicom_gradcoil.py` | header inventory of a DICOM folder (frame layout, trigger times) and the scanner / gradient-coil identification of Tab. 4.1 |
| `readvtk.py` | minimal reader for the legacy VTK file, used by the scripts above |

The repository contains the code only. The outputs of the thesis runs (the run records `run_config.json`, the figures, the ParaView state files and the VTK/NPZ velocity fields) are not kept here because of their size; the run records and ParaView states are available from the author on request. The DICOM datasets (phantom and volunteer data) are not distributed. The manufacturer's coefficient files (`.grad`) are not included either: they are the vendor's property and are obtained from the scanner or from the vendor on request; the toolbox reads them as they come.

## Requirements

Python 3.10 or newer with `numpy`, `scipy`, `matplotlib` and `pydicom`; `tkinter` (part of the standard Python installation) for the graphical file picker.

```
pip install numpy scipy matplotlib pydicom
```

## Usage

**Graphical picker** – start the script without arguments; a dialog asks for the coefficient file, the DICOM folder, the magnitude series, the output folder, the dataset preset, the VENC, the velocity-axis mapping, the analysis box, the simulation mode and the frame layout (the settings used for the thesis runs are listed in Table 3.4 of the thesis):

```
python 4D_MRI_GNL_correction.py
```

**Command line** – all options of the dialog are available as arguments (`python 4D_MRI_GNL_correction.py --help`). The commands used for the thesis datasets:

```
# 3 T pipe-flow phantom (steady flow, VENC 7 cm/s); the magnitude series is the smaller folder of the same acquisition
python 4D_MRI_GNL_correction.py --grad coeff_skyra.grad.txt --dicom <pipe velocity series folder> \
    --magnitude <pipe magnitude series folder> --preset pipe_flow --venc 0.07 --sim-mode 1 --out <output folder>

# 7 T idealised aneurysm phantom, series 11 (VENC 150 cm/s; the export carries no VENC, so it is given explicitly)
python 4D_MRI_GNL_correction.py --grad coeff_SC72.grad.txt --dicom <series 11 folder> --magnitude <series 10 folder> \
    --preset ideal_phantom --venc 1.5 --sim-mode 2 --direction-permute yxz --direction-signs=1,1,-1 \
    --roi-box 40,104,0,223,0,87 --out <output folder>

# the same acquisition as the post-processed export DIV11-9, whose frames are stored block-wise
python 4D_MRI_GNL_correction.py --grad coeff_SC72.grad.txt --dicom <DIV11-9 folder> --magnitude <series 10 folder> \
    --preset ideal_phantom --venc 1.5 --sim-mode 2 --direction-permute yxz --direction-signs=1,1,-1 \
    --roi-box 40,104,0,223,0,87 --directions-blocked --out <output folder>

# 7 T circle of Willis, in vivo (VENC 90 cm/s)
python 4D_MRI_GNL_correction.py --grad coeff_SC72.grad.txt --dicom <in vivo folder> --magnitude <multiframe magnitude file> \
    --preset in_vivo_head --venc 0.9 --sim-mode 2 --direction-permute yxz --direction-signs=-1,-1,1 --out <output folder>

# coil characterisation and synthetic verification only (no DICOM)
python 4D_MRI_GNL_correction.py --grad coeff_IMPULSE.grad.txt --preset ideal_phantom --out <output folder>
```

Key options:

| Option | Meaning |
|---|---|
| `--preset {pipe_flow, ideal_phantom, in_vivo_head}` | masking and figure settings for the type of dataset |
| `--venc M_S` | VENC of the acquired data in m/s; overrides the auto-detection from file names and DICOM tags. Velocities scale with it, so give it whenever the export does not carry it (the run stops with an error if none is found) |
| `--magnitude DIR_OR_FILE` | magnitude series (folder, or the single multiframe file of an export) for the signal mask |
| `--roi-box X0,X1,Y0,Y1,Z0,Z1` | restrict the analysis set (statistics only) to a voxel box; the correction is still applied to every voxel |
| `--direction-permute`, `--direction-signs` | order and sign of the three stored velocity components relative to the patient axes; needed for exports that lack the DICOM velocity-direction tags (see thesis, Section 3.6.1). The mapping is recorded in the run record. |
| `--directions-blocked` | the three velocity directions of a cardiac series are stored block-wise (all phases of direction 1, then 2, then 3) behind headers that kept the interleaved trigger times, as in a post-processed export; default is the scanner's interleaved order. The layout used is written to the run record (`runtime/dicom_frame_layout`). |
| `--snr-db`, `--sim-venc`, `--sim-nt`, `--sim-mode`, `--flow-pattern` | settings of the synthetic verification |
| `--no-npz` | skip the large `.npz` export |

Every run writes `run_config.json`, a complete record of the settings, geometry checks, self-test results and statistics of that run; the run records of the thesis runs reproduce every number reported in the thesis.

## Method in brief

- Λ(r) is obtained by central differences of the modelled field of each coil (Equation 3.4 of the thesis); the field is clamped at the calibration radius R₀ of the coefficient file and all statistics are restricted to r ≤ R₀.
- The inverse is applied through a singular-value gate (σ_min > 10⁻⁴) and an adaptive Tikhonov regularisation that is inactive for well-conditioned voxels (κ ≤ 2) and reaches λ = 0.05 at κ ≥ 10.
- The correction is multiplicative; additive phase errors (eddy currents, concomitant fields) are not addressed and must be removed separately.

## Citation

If you use this toolbox, please cite the thesis and the archived release (see `CITATION.cff`):

> Jaison J. *Open-Source Preprocessing for Gradient Nonlinearity Correction in 4D Flow MRI.* Master thesis, Otto von Guericke University Magdeburg, 2026. Toolbox v1.0.1: https://github.com/jainajaison13-source/4D_MRI_GNL_correction, DOI 10.5281/zenodo.23348757 (all versions).

Underlying method: Markl M, Bammer R, Alley MT, et al. Generalized reconstruction of phase contrast MRI: analysis and correction of the effect of gradient field distortions. *Magn Reson Med* 2003;50:791–801.

## Licence

MIT License – see `LICENSE`.
