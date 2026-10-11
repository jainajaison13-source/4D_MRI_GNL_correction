#!/usr/bin/env python3
r"""
4D Flow MRI - Gradient Nonlinearity Correction Toolbox
======================================================
Open-source preprocessing for gradient nonlinearity (GNL) correction in
phase-contrast / 4D flow MRI.

Velocity encoding in phase-contrast MRI assumes spatially linear magnetic
field gradients. Real gradient systems are nonlinear, which introduces
spatially dependent errors in both the magnitude and direction of the
encoded velocity. This toolbox characterises the gradient field from the
manufacturer's spherical-harmonic coefficient file and applies a voxel-wise
correction to phase-contrast velocity data.

Pipeline
--------
  1. Parse the manufacturer gradient coefficient file (.grad / .txt) and
     build the spherical-harmonic expansion of the gradient field.
  2. Construct the voxel-wise 3x3 distortion matrix Lambda(r) mapping the
     true velocity to the encoded (measured) velocity.
  3. Invert Lambda(r) per voxel to recover the corrected velocity field.
  4. Validate the correction against an analytical/synthetic ground truth in
     a controlled 4D simulation (steady Poiseuille tube, pulsatile vessel).
     This is NUMERICAL VERIFICATION of the reconstruction mathematics —
     velocities are sampled at voxel centres; no intra-voxel signal
     integration, complex MR signal, or k-space model is simulated — not an
     MRI acquisition simulation. Physical validation comes from the acquired
     phantom data (step 5) and its Lambda-independent flux checks.
  5. Apply the correction to reconstructed DICOM velocity volumes, with
     Lambda re-evaluated on the DICOM acquisition grid.

Component map
-------------
  GradientFieldParser - reads the coefficient file, builds the SH expansion
  GradientFieldEvaluator - evaluates the field and constructs Lambda(r)
  SimulationEngine4D - synthetic 4D velocity fields and forward model
  DicomLoader - Siemens Enhanced / Multiframe 4D-flow DICOM I/O
  VelocityCorrector - voxel-wise Lambda^-1 gradient-nonlinearity correction
  ValidationEngine - RMSE, MAE, Pearson, Bland-Altman, bootstrap CIs
  PlottingEngine - diagnostic and validation figures

SCOPE
-----
This script corrects GRADIENT NONLINEARITY only, per the thesis task
description ("a generalized velocity reconstruction framework ... that
corrects spatially varying first gradient moments on a voxel-wise basis").

Eddy-current and concomitant-field (Maxwell) phase corrections are a
different class of error -- ADDITIVE phase offsets, not the MULTIPLICATIVE
velocity distortion Lambda -- and are deliberately out of scope: no code
here measures, fits, subtracts or reports them. A dataset carrying such
offsets carries them through this pipeline unchanged; that is by design,
not an oversight, and must be stated as a scope limitation when reporting
absolute velocities. Vendors normally apply both during reconstruction.

NOTE ON THE WORD "MAXWELL": the Maxwell DISTRIBUTION appearing in the noise
model below is the statistics of |v| = sqrt(vx^2+vy^2+vz^2) for Gaussian
components -- an unrelated use of the same name, and load-bearing for the
noise floor and the vessel threshold. It is not the concomitant field.

All quantitative metrics are restricted to voxels inside the spherical-harmonic
calibration sphere (r <= R0), where the field expansion is valid.

Output figures
--------------
  fig01 - Lambda(r) distortion / correction matrix
  fig02 - velocity magnitude percentage error, before vs after correction
  fig03 - spatially varying VENC(r) field
  fig04 - summary validation metrics (RMSE / MAE, before vs after)
  fig05 - DICOM correction diagnostic (only when a DICOM directory is given)

Self-tests
----------
  T1  - identity Lambda round-trip
  T2  - non-trivial (non-symmetric) Lambda round-trip
  T3  - SH evaluator vs the closed-form derivative of a known harmonic
  T3b - regression: primary linear term is not counted twice
  T4  - Poiseuille peak-velocity check on the discrete grid
  T5  - model-mismatch robustness (correct with perturbed coefficients)

Scope and limitations of the validation
---------------------------------------
Read this before quoting any number from the synthetic pipeline.

1. The synthetic forward model and the correction use the SAME Lambda(r).
   Inverting the operator that was just applied is a check of the
   implementation and its numerical conditioning, not of the physical
   model. Tests T1/T2 are therefore circular BY DESIGN and must not be
   presented as evidence that the correction works on a scanner.
   T5 is the non-circular test: it distorts with one coefficient set and
   corrects with coefficients perturbed by 2, 5 and 10 % (one random
   per-coefficient Gaussian draw per level), so it measures robustness to the
   realistic case of an imperfect calibration file. Quote T5, and the
   DICOM/phantom results, as the evidence for the method.

2. Lambda is built from the manufacturer's spherical-harmonic expansion,
   which is defined only inside the calibration sphere r <= R0. Outside
   it the radial factor is clamped, so Lambda collapses to the identity -
   an artefact that mimics a perfectly linear gradient. All metrics are
   masked to r <= R0 - eps; see GradientFieldEvaluator.inside_r0_mask.

3. The simulated Poiseuille tube is enlarged to 30 mm radius so that it
   spans >= 6 voxels on the working grid. The velocities are realistic;
   the implied volumetric flow rate is not, and is reported only for
   internal consistency between figures.

4. Eddy-current and concomitant-field (Maxwell) phase offsets are NOT
   corrected and NOT measured anywhere in this toolbox (see SCOPE).
   Reconstructed velocity DICOM also lacks the gradient-waveform and
   timing information the concomitant-field correction would require.

Inputs (selected via the GUI)
-----------------------------
  - gradient coefficient file (.grad / .txt)
  - DICOM directory (optional - enables real-data correction)
  - output directory for figures and logs

Equation map — which code implements which published equation
--------------------------------------------------------------
  GradientFieldEvaluator._bz_field        Markl 2003 Appendix Eq. [A1]
                                          (Romeo & Hoult 1984 field model)
  GradientFieldEvaluator.compute_lambda   Markl 2003 Eqs. [1]-[2]
                                            Lambda_ji = G_ji,model / G_i,ideal
  .strength_deviation                     Markl 2003 Eq. [11], d_i
  .angular_deviation_deg                  Markl 2003 Results, theta_i
  SimulationEngine4D.apply_forward_distortion
                                          Markl 2003 Eqs. [4]-[6]
                                            v_meas = Lambda^T v_true
  VelocityCorrector.correct_4d            Markl 2003 Eq. [7] / [A5]
                                            v = [Lambda^T]^-1 dPhi
   - Tikhonov branch                      Markl 2003 Appendix Eqs. [A6]-[A7]
                                            (SVD/regularisation of ill-posed
                                             inversion); kappa->lambda ramp is
                                             a custom heuristic
  PlottingEngine.plot_spatially_varying_venc
                                          VENC(r) = VENC / d_i, from Eq. [11]
  SimulationEngine4D.add_phase_noise      sigma_v = sqrt(2)*VENC/(pi*SNR)
                                          (noise of a phase difference; see
                                          the add_phase_noise docstring)
  generate_poiseuille_4d / self-test T4   Hagen-Poiseuille reference and the
                                          10% pixel/diameter criterion
                                          (Fukuyama 2017; Watanabe 2019)

Known deviations from the published methods
-------------------------------------------
  1. Single-direction encoding (Markl Eqs. [10]-[12]) is not implemented;
     this toolbox assumes three-directional acquisition, which is what the
     generalized reconstruction of Eq. [7] requires anyway.
  2. Voxel coordinates are assumed already spatially unwarped, as Markl
     Eq. [2] requires. See compute_lambda. For ND (no vendor distortion
     correction) data Lambda is therefore evaluated at the nominal, not the
     true, voxel position.
  3. The adaptive Tikhonov branch (kappa -> lambda ramp) is a custom
     heuristic, not part of Markl et al.; it is inactive wherever
     kappa <= 2 (see SimulationConfig.tikhonov_kappa_low).
  (Eddy-current and concomitant-field corrections are not deviations but
  out of scope; see SCOPE above.)

References
----------
Bernstein MA, Zhou XJ, Polzin JA, King KF, Ganin A, Pelc NJ, Glover GH.
    Concomitant gradient terms in phase contrast MR: analysis and
    correction. Magn Reson Med 1998;39:300-308.
    -> background only: defines the concomitant (Maxwell) field, which is
       OUT OF SCOPE here (see SCOPE). Cited to delimit what this toolbox
       does not do.

Markl M, Bammer R, Alley MT, Elkins CJ, Draney MT, Barnett A, Moseley ME,
    Glover GH, Pelc NJ. Generalized reconstruction of phase contrast MRI:
    analysis and correction of the effect of gradient field distortions.
    Magn Reson Med 2003;50:791-801.
    -> the method implemented here. Lambda_ji definition Eqs. [1]-[2];
       spatially dependent first moment Eq. [4]; encoding Eqs. [5]-[6];
       generalized reconstruction Eq. [7]; single-direction Eqs. [10]-[12];
       SH expansion Appendix Eq. [A1]; SVD treatment Eqs. [A6]-[A7].

Lorenz R, Bock J, Snyder J, Korvink JG, Jung BA, Markl M. Influence of
    eddy current, Maxwell and gradient field corrections on 3D flow
    visualization of 3D CINE PC-MRI data. Magn Reson Med 2014;72:33-40.
    -> the GNL portion is the relevant one here: evaluation of the gradient
       field correction on 3D CINE PC-MRI, and the demonstration that these
       errors grow nonlinearly away from isocentre. Its eddy/Maxwell stages
       are outside this toolbox's scope (see SCOPE).

Watanabe T, Isoda H, Fukuyama A, et al. Accuracy of the flow velocity and
    three-directional velocity profile measured with three-dimensional
    cine phase-contrast MR imaging: verification on scanners from
    different manufacturers. Magn Reson Med Sci 2019;18:265-271.
    -> Hagen-Poiseuille reference profile for phantom validation;
       cross-vendor accuracy context; restates the 10% pixel-to-diameter
       criterion of Fukuyama et al. (their ref 16).

Fukuyama A, Isoda H, Morita K, et al. Influence of spatial resolution in
    three-dimensional cine phase contrast magnetic resonance imaging on the
    accuracy of hemodynamic analysis. Magn Reson Med Sci 2017;16:311-316.
    -> origin of the "peak-velocity error < 10% when pixel size / inner
       diameter < 10%" criterion cited by Watanabe et al. 2019.

Romeo F, Hoult DI. Magnet field profiling: analysis and correcting coil
    design. Magn Reson Med 1984;1:44-65.
    -> spherical-harmonic field model underlying the coefficient files.

Markl M, Frydrychowicz A, Kozerke S, Hope M, Wieben O. 4D flow MRI.
    J Magn Reson Imaging 2012;36:1015-1036.
    -> aortic waveform shape and helical/swirling flow used by
       cardiac_waveform and generate_pulsatile_vessel_4d.

Efron B, Tibshirani RJ. An Introduction to the Bootstrap. Chapman & Hall,
    New York, 1993. (Chapter 13, bootstrap percentile intervals.)
    -> resampling scheme and 95% confidence intervals in
       ValidationEngine.rmse_bootstrap_ci.
"""

# ─────────────────────────────────────────────────────────────────────────────
# Standard library
# ─────────────────────────────────────────────────────────────────────────────
import os
import re
import sys
import copy
import json
import time
import shutil
import warnings
import traceback
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timezone
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

# Partial run record, so an aborted run still leaves a diagnosable file.
# main() fills this in as soon as run_config.json first exists; the top-level
# handler at the bottom uses it to append an "aborted" section.
_RUN_STATE = {"path": None, "log": None}

# ─────────────────────────────────────────────────────────────────────────────
# Scientific stack
# ─────────────────────────────────────────────────────────────────────────────
import numpy as np
from numpy.linalg import inv as np_inv
from scipy.special import lpmv, factorial as sp_factorial
from scipy.stats import pearsonr
import matplotlib
matplotlib.use("Agg")                    # Windows-safe non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm

# ─────────────────────────────────────────────────────────────────────────────
# ┌─ USER SETTINGS — edit these for your DICOM dataset ───────────────────────┐
# │ These two values are NOT reliably stored in 4D-flow DICOM headers, so │
# │ set them here to match your scan/phantom. Leave VENC = None to let the │
# │ loader auto-detect (filename 'venc..' → tag → CSA → text → error). │
# └───────────────────────────────────────────────────────────────────────────┘
DICOM_VENC_OVERRIDE_MS: Optional[float] = None     # e.g. 1.50  for a 150 cm/s scan;
                                                    # or 0.07 for a pipe rig at 7 cm/s

# ┌───────────────────────────────────────────────────────────────────────────┐
# │ Strict VENC handling. True (default): when VENC is found in neither       │
# │ filename nor header, RAISE rather than guess, forcing you to supply       │
# │ DICOM_VENC_OVERRIDE_MS explicitly. Quantitative velocities scale linearly │
# │ with VENC, so a silently invented value corrupts every number downstream. │
# │                                                                           │
# │ This was False (silent fallback to 0.90 m/s) in an earlier version. That  │
# │ default was itself dataset-specific — 0.90 m/s is correct for the         │
# │ Circle-of-Willis protocol this pipeline was developed on, and silently    │
# │ wrong for anything else. It stayed hidden on a pipe-flow rig at VENC      │
# │ 0.07 m/s because the header's CSA private tag (0x0019,0x1018) held        │
# │ 11700/17600 — outside the tag's own 5-500 cm/s plausibility guard below,  │
# │ so every detection path failed and the pipeline fell back to 0.90 m/s     │
# │ with only a printed warning, an easy line to miss. A public toolbox run   │
# │ against an unknown DICOM folder must not be able to do that: set False    │
# │ only if you have independently confirmed the fallback value is correct    │
# │ for every dataset the run will ever see.                                  │
# └───────────────────────────────────────────────────────────────────────────┘
DICOM_REQUIRE_VENC: bool = True

# ┌───────────────────────────────────────────────────────────────────────────┐
# │ Console verbosity for the simulation-vs-data SNR comparison.              │
# │                                                                           │
# │ True  — full banner (developer mode: use it the FIRST time you run a new  │
# │         dataset, so the mismatch cannot be missed).                       │
# │ False — one factual line: "noise floor implies SNR ≈ N dB for this        │
# │         dataset (simulation figures: M dB — see the SNR sweep)".          │
# │                                                                           │
# │ Setting this False quietens the CONSOLE only. The measured SNR, the       │
# │ measured sigma_v, its source and the simulation SNR are always written to │
# │ run_config.json, because they are properties of the ACQUISITION and the   │
# │ run must stay self-documenting.                                           │
# │                                                                           │
# │ Running the simulation at a higher SNR than the acquired data is a        │
# │ legitimate, defensible choice: fig02/fig04 exist to show the SPATIAL      │
# │ STRUCTURE of the systematic GNL error, and at the in-vivo noise level     │
# │ (sigma_v ~ 4 cm/s vs a ~0.8 cm/s systematic error) that structure is      │
# │ buried. What must NOT happen is quoting the resulting % improvement as if │
# │ it described the acquired data. Report instead: (1) the noise-free row,   │
# │ which is SNR-independent by construction, and (2) the full SNR sweep      │
# │ table, which shows the noise dependence including the unfavourable rows.  │
# │ Both are printed by this pipeline regardless of this flag.                │
# └───────────────────────────────────────────────────────────────────────────┘
SNR_MISMATCH_VERBOSE: bool = False

# ┌───────────────────────────────────────────────────────────────────────────┐
# │ Velocity-encoding direction order, for interleaved 3-direction series.    │
# │                                                                           │
# │ Such a series stores the three encoding directions as consecutive frames, │
# │ and nothing in the frame order itself says which physical axis each one   │
# │ measures. Getting this wrong permutes the velocity components, which      │
# │ matters because Λ is ANISOTROPIC — Λxx, Λyy and Λzz differ, so a          │
# │ permuted component is corrected with the wrong matrix element.            │
# │                                                                           │
# │ None  → auto-detect from the Siemens SequenceName suffix (_rl, _ap, _fh,  │
# │         _in, _tp, and the logical _r / _p / _s), falling back to           │
# │         acquisition order if the tag is absent. This is the default and   │
# │         is right for most data.                                           │
# │ "xyz" → keep acquisition order (frame 0→vx, 1→vy, 2→vz), no detection.    │
# │ "zxy" → frame 0 is vz, frame 1 is vx, frame 2 is vy. Any permutation of   │
# │         "xyz" works; use this to correct a known mis-assignment.          │
# └───────────────────────────────────────────────────────────────────────────┘
DICOM_DIRECTION_ORDER: Optional[str] = None

# ┌───────────────────────────────────────────────────────────────────────────┐
# │ Per-component velocity SIGN flips, applied AFTER any axis permutation.    │
# │ None → keep the signs as loaded. Or a 3-tuple of ±1, e.g. (+1, -1, +1)    │
# │ to negate vy only.                                                        │
# │                                                                           │
# │ Why this exists: the direction detection maps 'rl'/'lr' (etc.) onto the   │
# │ same axis, but Siemens defines a POLARITY per encoding that the suffix    │
# │ does not carry. A wrong sign is invisible in |v| maps, yet it corrupts    │
# │ the cross-term mixing of the Λ⁻ᵀ correction (the off-diagonal terms of    │
# │ Λ couple the components with signed weights).                             │
# │                                                                           │
# │ How to verify on a phantom (one ParaView session):                        │
# │   1. Open velocity_field.vtk, threshold on 'anatomy_magnitude'.           │
# │   2. Add a Glyph filter on 'velocity_measured' in the inlet tube.         │
# │   3. The arrows must point the way the pump actually pumps. If a          │
# │      component is mirrored, set the corresponding entry to -1 here        │
# │      and re-run. In vivo: check arterial flow is head-ward (+z in LPS     │
# │      is Superior) in the carotids/basilar.                                │
# └───────────────────────────────────────────────────────────────────────────┘
# Verified against intracranial anatomy on the Circle-of-Willis dataset
# (2026-08): the previous value (1, 1, -1) put ALL THREE components in the
# wrong direction. Six textbook checks — each MCA flowing laterally, ICA
# terminus and basilar ascending, ACA anterior, PCA posterior — agreed with
# the data only 6-33% of the time; after flipping all three they agree
# 72-96%. The export carries neither Velocity Encoding Direction (0018,9090)
# nor the MR Velocity Encoding Sequence (0018,9197), so the convention
# cannot be read from the header and has to be set here. Re-verify with
# verify_signs.py, or DICOM_VERIFY_SIGNS_ANATOMICALLY below, after changing
# dataset — a wrong sign is invisible in |v|, |dv| and the correction angle.
#
# DEFAULT IS None (no sign change). The values found for the thesis datasets
# are recorded here as PROVENANCE, not applied: supply them per run with
# --direction-signs (see "AXIS FLAGS ARE PER-RUN" below).
# Both 7 T datasets need the permutation "yxz" AND a relative sign flip of
# vz (both MeVisLab exports of the same 7 T system):
#   Circle of Willis (xpace_fs4DMRI): --direction-permute yxz
#                                     --direction-signs=-1,-1,1
#       vessel-axis alignment median |cos(v, vessel axis)| = 0.971 with this
#       mapping vs 0.625 as loaded; anatomy 70-94 % agreement on all six
#       textbook directions (MCA, ICA terminus, basilar, ACA, PCA).
#   Aneurysm phantom, series 11 / DIV11-9: --direction-permute yxz
#                                     --direction-signs=1,1,-1  (supply tube)
# (1,1,-1) and (-1,-1,1) differ only by a GLOBAL polarity flip, which
# commutes with the correction (Λ⁻ᵀ(−v) = −Λ⁻ᵀv), so for the phantom either
# gives identical statistics. Relative to NO sign change they are NOT
# equivalent: flipping vz relative to vx, vy changes the cross-term mixing
# and every magnitude/angle result. The 3 T pipe series need neither flag.
DICOM_DIRECTION_SIGNS: Optional[Tuple[int, int, int]] = None

# ┌───────────────────────────────────────────────────────────────────────────┐
# │ Automatic ANATOMICAL verification of the sign convention (brain only).    │
# │                                                                           │
# │ The sign of a phase-contrast velocity cannot be recovered from the data   │
# │ alone: it is a vendor convention carried by the DICOM Velocity Encoding   │
# │ Direction attribute (0018,9090) / MR Velocity Encoding Sequence           │
# │ (0018,9197). Research exports frequently strip those tags, in which case  │
# │ DICOM_DIRECTION_SIGNS must be supplied by hand — and a wrong sign is      │
# │ INVISIBLE in every magnitude-based figure, because |v|, |dv| and the      │
# │ angle between v_meas and v_corr are all unchanged when v -> -v.           │
# │                                                                           │
# │ Set this True for a Circle-of-Willis / intracranial dataset and the run   │
# │ checks the loaded velocities against six textbook facts:                  │
# │     MCA flows laterally away from the midline on each side                │
# │     ICA terminus and basilar ascend  (+z, superior)                       │
# │     ACA runs anteriorly (-y), PCA posteriorly (+y)                        │
# │ and reports the fraction of vessel voxels that agree. Consistent          │
# │ disagreement means the signs are inverted, and the recommended            │
# │ replacement tuple is printed. Nothing is changed automatically — the      │
# │ anatomy is a prior, not a measurement, so the decision stays with you.    │
# │ Leave False for phantoms, where these anatomical facts do not apply.      │
# └───────────────────────────────────────────────────────────────────────────┘
DICOM_VERIFY_SIGNS_ANATOMICALLY: bool = True

# ┌───────────────────────────────────────────────────────────────────────────┐
# │ Post-load component PERMUTATION, applied to the FINAL loaded array —      │
# │ i.e. AFTER the loader's own frame→axis assignment (header detection or    │
# │ DICOM_DIRECTION_ORDER), and BEFORE DICOM_DIRECTION_SIGNS.                 │
# │                                                                           │
# │   "yxz" → the corrected vx is the component the loader placed in the y    │
# │           slot, corrected vy is the loader's x slot, vz unchanged.        │
# │   None  → leave the loaded assignment alone.                              │
# │                                                                           │
# │ This is deliberately separate from DICOM_DIRECTION_ORDER. That flag       │
# │ REPLACES the header detection inside one loader and is expressed in       │
# │ acquisition-frame order; this one repairs whatever came out, works for    │
# │ every input format, and is expressed in the axis labels you can actually  │
# │ inspect in the exported VTK.                                              │
# │                                                                           │
# │ WARNING: do not use both to fix the same problem — fixing the frame       │
# │ order upstream and then permuting again here applies the swap twice.      │
# │                                                                           │
# │ ── Why "yxz" and (1, 1, -1) were found for the Series11 phantom ──────── │
# │ Determined from the exported field, not assumed. The supply line is a     │
# │ ~3.5 mm bore tube, so flow in it must be parallel to its centreline —     │
# │ a hard constraint needing no header. Measured on that tube:               │
# │   · centreline tilt ratios      |x/y| = 0.037   |z/y| = 0.287             │
# │   · loaded component ratios     |#1/#0| = 0.046  |#2/#0| = 0.290          │
# │     → loaded #0 is physical y (it dominates a tube running along y),      │
# │       #2 is z (0.290 vs 0.287), #1 is x (0.046 vs 0.037). Assigning #2    │
# │       to x instead would demand 0.290 ≈ 0.037, an 8× mismatch.            │
# │   · flux continuity ∮v·dA along the tube is constant only when #0 sits    │
# │     in the y slot (|Q| = 0.50, CV 0.23); as loaded it is 0.02 — a tube    │
# │     carrying 50 cm/s with no net flux, which is impossible.               │
# │   · signed alignment to the centreline: yxz with vz negated gives         │
# │     median cos = 0.9994 (worst segment 0.982), against 0.943 for the      │
# │     best alternative and −0.16 as loaded.                                 │
# │                                                                           │
# │ Two residual ambiguities, neither affecting any reported statistic:       │
# │   1. GLOBAL polarity. Flipping all three signs commutes with the linear   │
# │      correction — Λ⁻ᵀ(−v) = −Λ⁻ᵀ(v) — so |v|, |Δv| and every metric are   │
# │      identical. It only reverses the arrows in ParaView. Glyph the inlet  │
# │      tube: if flow runs against the pump, use (-1, -1, 1) instead.        │
# │   2. The x sign is NOT determined here: x carries 4.6% of the tube flow,  │
# │      so both choices align to ~0.99. Take it from the protocol if the     │
# │      cross-term mixing at large |x| matters for your numbers.             │
# │                                                                           │
# │ These are properties of THIS acquisition, so they are deliberately NOT    │
# │ part of any dataset preset — a preset switch must not silently rewrite    │
# │ them. Set them back to None / None for a different series.                │
# └───────────────────────────────────────────────────────────────────────────┘
#
# DEFAULT IS None (keep the loader's assignment). "yxz" was first derived on
# the aneurysm-phantom supply tube (series 11 / DIV11-9, evidence above) and
# was later confirmed independently on the Circle-of-Willis export (vessel-
# axis alignment, see DICOM_DIRECTION_SIGNS). It is recorded as PROVENANCE
# only; supply it per run with --direction-permute yxz. Verify it for every
# new export — a wrong permutation leaves |v| unchanged but applies the
# wrong rows of Λ.
DICOM_DIRECTION_PERMUTE: Optional[str] = None

# ┌───────────────────────────────────────────────────────────────────────────┐
# │ SCOPE GUARD for the two flags above.                                      │
# │                                                                           │
# │ DICOM_DIRECTION_PERMUTE and DICOM_DIRECTION_SIGNS describe how ONE series │
# │ was encoded. They are deliberately not part of any dataset preset (a      │
# │ preset switch must not rewrite them) — but that also means they would     │
# │ otherwise apply to EVERY dataset you point the tool at, including the     │
# │ in-vivo Circle-of-Willis series, whose axes were never shown to need      │
# │ them. Applying a phantom's swap to the head data would silently corrupt   │
# │ it: |v| is unchanged, so nothing downstream looks wrong.                  │
# │                                                                           │
# │ This is a case-insensitive substring. The permutation and the sign flips  │
# │ are applied ONLY when it appears in the DICOM path, the dataset label or  │
# │ the output directory; otherwise they are skipped and the reason is        │
# │ printed. Set to None to apply them unconditionally (rarely what you       │
# │ want), or "" to disable the axis flags entirely.                          │
# │                                                                           │
# │ The failure mode is deliberately the safe one: rename the folder and the  │
# │ flags stop applying — loudly, in the console — rather than being applied  │
# │ to a series they were never verified on.                                  │
# └───────────────────────────────────────────────────────────────────────────┘
# Anchor this on the DICOM path, NOT on the output folder name.
#
# The scope is matched against dicom_dir + DICOM_DATASET_LABEL + output_dir.
# The previous value "Series11" appears in neither phantom's DICOM path
# (study_3_series_DIV11-9_..., Velocityseries_11_...) — it matched only
# because the OUTPUT folders were named "…Series11-9…" and "…Series11…".
# That is fragile in both directions:
#   - renaming or clearing the output folder silently switches the axis
#     permutation OFF, changing the component mapping with no warning and no
#     visible change to |v|;
#   - a differently named output folder for another series silently switches
#     it ON, applying a mapping derived elsewhere (which is how series 10+11
#     inherited DIV11-9's permutation).
# "DIV11-9" occurs in that series' DICOM directory, so the match now follows
# the data rather than a folder name you choose.
# The axis flags (DICOM_DIRECTION_PERMUTE / DICOM_DIRECTION_SIGNS) encode a
# component mapping verified for ONE acquisition, so they must never be
# inherited by another. What this setting does, as the code actually behaves:
#
#   "some string" -> the flags are applied ONLY when that string appears in
#                    the DICOM path (or dataset label / output folder); every
#                    other dataset is processed with axes exactly as loaded.
#   ""            -> the flags are disabled for every dataset.
#   None          -> the flags are applied UNCONDITIONALLY to every dataset.
#
# NOTE: an earlier version of this comment said "None = NOT applied". That was
# the opposite of the behaviour — the run console prints "Axis flags applied
# UNCONDITIONALLY (DICOM_DIRECTION_SCOPE = None)" and applies them. With the
# defaults below, a new acquisition (a pipe rig, say) therefore inherits the
# Circle-of-Willis mapping unless you either set a scope string from the CoW
# DICOM path, or set DICOM_DIRECTION_PERMUTE and DICOM_DIRECTION_SIGNS to None
# for that run.
#
# ── AXIS FLAGS ARE PER-RUN (since v1.0) ─────────────────────────────────────
# DICOM_DIRECTION_PERMUTE and DICOM_DIRECTION_SIGNS now default to None, so a
# dataset is processed with its axes exactly as loaded unless a mapping is
# asked for explicitly:
#     --direction-permute yxz  --direction-signs=1,1,-1
# Values given on the command line apply to that run only, bypass the scope
# guard below, and are recorded in run_config.json under
# runtime.axis_flags_source = "cli". Values edited into this file instead are
# still subject to DICOM_DIRECTION_SCOPE (source "file"). With neither, the
# source is "none" and no permutation or sign flip is applied.
# To reproduce a run made with the old defaults, read
# dicom.axis_flags.permute_applied / signs_applied from its run_config.json
# and pass the same values on the command line.
DICOM_DIRECTION_SCOPE: Optional[str] = None
# Where this run's axis flags came from: "cli" (command line), "dialog",
# "file" (edited into this script) or "none". Set in main().
AXIS_FLAGS_SOURCE: str = "none"

# Derive the slice-to-slice step from the acquisition GEOMETRY
# (SpacingBetweenSlices, or consecutive ImagePositionPatient differences
# projected onto the slice normal) instead of SliceThickness.
#
# SliceThickness is the excitation width; with gapped or overlapping slices
# the geometric step differs, and using the wrong one stretches every z
# position, so Λ is evaluated at the wrong distance from isocentre. For a
# contiguous 3D acquisition the two are identical and this changes nothing.
#
# It is a flag because it is the ONE correction in this file that can move
# already-published numbers: if a dataset's two values differ, results
# computed before this existed used the wrong z. The console says explicitly
# whether the values differed, so you can check whether any given dataset is
# affected. Set False only to reproduce a pre-existing run byte-for-byte;
# leave True for anything you intend to defend.
DICOM_GEOMETRIC_SLICE_SPACING: bool = True

# Refuse to proceed when the input does not look like phase-contrast velocity
# data — for example a MAGNITUDE series selected by mistake. Such a run
# produces statistics and figures that look entirely normal but mean nothing,
# so failing loudly is safer than continuing. Set False to downgrade the checks
# to warnings.
DICOM_STRICT_INPUT_CHECK: bool = True

# ── Temporal-coherence gate: separates FLOWING fluid from STATIC fluid ───────
#
# A speed threshold alone cannot do this. Both compartments of a flow phantom
# are bright water, so the magnitude mask keeps both, and the vessel cut is
# applied to mean_t(|v|) — the mean of the MAGNITUDE, which RECTIFIES noise:
# in static fluid each component is ~N(0, sigma) and |v| follows a Maxwell
# distribution with mean 1.6*sigma, a positive number no averaging removes.
# On the ideal phantom (sigma = 3.1 cm/s) that puts the static water sphere at
# a mean |v| of ~5 cm/s against a 7.9 cm/s cut, so its upper tail passes and
# the "vessel" mask filled with 189,309 voxels of stationary water.
#
# Temporal coherence separates them, because the two behave oppositely under
# time-averaging:
#
#     coherence = |mean_t(v)| / mean_t(|v|)
#
#   static fluid : the direction is random each cardiac phase, so the VECTOR
#                  average shrinks by ~1/sqrt(Nt) while the MAGNITUDE average
#                  does not -> coherence ~ 1/sqrt(Nt)  (0.30 at Nt = 11)
#   real flow    : direction is consistent across the cycle -> coherence ~ 1
#
# 0.50 sits between the two and is safe for pulsatile flow that accelerates
# and decelerates without reversing. Lower it if the flow REVERSES during the
# cycle, since a reversing vessel partially cancels its own time average.
#
# None disables the gate. Leave it off for in-vivo brain data: parenchyma is
# already handled by the magnitude and brain masks, and a coherence cut there
# would discard genuine slow or pulsatile vessels.
DICOM_FLOW_COHERENCE_MIN: Optional[float] = None

# ── Flow-selected SUBSET, reported alongside the primary analysis set ────────
# Set a coherence value (0.5 is a good default) to additionally report the
# correction over voxels that actually carry flow. Unlike
# DICOM_FLOW_COHERENCE_MIN this does NOT change the primary analysis set — it
# adds a second block, so one run yields both:
#
#   primary  : every magnitude/brain-masked voxel (includes slow vessels AND
#              parenchyma, whose apparent speed is rectified noise)
#   subset   : the temporally coherent voxels — real flow keeps its direction
#              across the cardiac cycle (coherence ~ 1), noise does not
#              (~1/sqrt(Nt)), so a SLOW but genuine vessel is kept while a
#              faster-looking noise voxel is not. This is the discriminator a
#              speed threshold cannot provide.
#
# Reporting both is the honest way to satisfy "include the slow vessels"
# without letting the headline number be dominated by stationary tissue.
# None disables the extra block.
DICOM_COHERENT_SUBSET_MIN: Optional[float] = 0.5

# Upper end of the fig05 velocity colour scale, in cm/s. None = automatic
# (99th percentile of the displayed voxels).
#
# The automatic rule fails on a phantom containing compartments with very
# different velocities: a small fast jet sets the scale and compresses the main
# chamber into the bottom tenth of the colormap, where everything looks dark
# blue regardless of structure. A single linear scale cannot show a 13 cm/s
# chamber and a 110 cm/s jet at once — cap it at whatever the region of
# interest actually reaches.
DICOM_FIG05_VMAX_CMS: Optional[float] = None
# Title label for the DICOM figure (fig05). Leave empty to auto-derive the
# label from the output directory name (recommended) so the figure always
# matches the dataset being corrected; set a string here to override.
DICOM_DATASET_LABEL:     str            = ""
# Optional: path to the paired MAGNITUDE series folder. Phase-contrast velocity
# in signal-void regions (air/background) is random noise spanning ±VENC, which
# swamps the real flow. If you give the magnitude folder here, the DICOM stage
# builds a signal mask (magnitude > 10% of max) so fig05 and the correction
# stats focus on the actual lumen/vessel instead of background noise.
# Selected at runtime via the GUI (optional); may also be set here as a default.
DICOM_MAGNITUDE_DIR:     Optional[str]  = None
# Multiframe (Circle-of-Willis / MeVisLab / xpace) frame ordering. Files without
# per-frame position metadata store the 884 = Nz×Nt frames either phase-major
# (frame = phase*Nz + slice) or slice-major (frame = slice*Nt + phase). Guessing
# wrong scrambles slice positions → Λ applied at the wrong distance from
# isocentre → sawtooth in the per-slice correction. "auto" detects the order
# from the (clean) magnitude pixel data; override with "phase_major" or
# "slice_major" only if you know the export convention.
DICOM_MULTIFRAME_ORDER:  str            = "auto"
# ═════════════════════════════════════════════════════════════════════════════
# DATASET PRESET — the ONE setting to change between datasets
# ═════════════════════════════════════════════════════════════════════════════
# The in-vivo head and the phantoms need genuinely different display and
# masking choices, and getting them crossed produces silently wrong figures
# (a brain mask erodes a thin pipe lumen; an arterial velocity floor hides slow
# phantom flow). Set this once per dataset; it overwrites the individual
# DICOM_* settings further down.
#
#   "in_vivo_head" — Circle of Willis / any brain 4D flow
#   "phantom"      — pipe flow, ideal phantom, steady flow rig
#   "manual"       — ignore the preset, use the individual settings as written
DATASET_PRESET: str = "ideal_phantom"

# Vessel threshold for the in-vivo correction statistics, in cm/s.
#   0.0   -> NO velocity gate. The analysis set is the MAGNITUDE/brain mask
#            alone, so every voxel with anatomical signal is included, however
#            slowly it flows. Use this when slow vessels must not be excluded.
#   value -> force that threshold in cm/s.
#   None  -> derive it from the measured noise floor as
#            DICOM_VESSEL_SIGMA_MULT × σ (with a VENC-relative lower bound).
#
# WHAT THE GATE WAS FOR — read before choosing 0.0. A magnitude mask separates
# TISSUE from AIR; it does not separate BLOOD from PARENCHYMA, because brain
# tissue is bright in magnitude too. In parenchyma the three velocity
# components are pure noise, and |v| = sqrt(vx²+vy²+vz²) RECTIFIES that noise
# into a positive speed: for Gaussian components of width σ the mean of |v| is
# sqrt(8/π)·σ ≈ 1.60·σ, which no amount of averaging removes. At the σ ≈ 4 cm/s
# noise floor of a 7 T CoW acquisition that is ≈ 6.5 cm/s of apparent speed in
# tissue that is not moving at all. Statistics over the unthresholded mask are
# therefore dominated by voxels whose velocity is noise, and |Δv| scales with
# the local speed, so the reported mean relative correction describes the
# correction applied to NOISE rather than to blood.
#
# If the goal is to INCLUDE SLOW VESSELS rather than to include parenchyma,
# the discriminator to use is temporal coherence, not speed: real flow keeps
# its direction across the cardiac cycle (coherence ≈ 1) while noise does not
# (coherence ≈ 1/sqrt(Nt)). A slow but genuine vessel passes; a faster-looking
# noise voxel does not. See DICOM_FLOW_COHERENCE_MIN.
#
# Whichever is chosen, the run reports how much of the analysis set is
# consistent with pure noise, so the number quoted in the thesis can be
# defended.
DICOM_VESSEL_THRESHOLD_CMS: Optional[float] = None
DICOM_VESSEL_SIGMA_MULT:    float           = 4.0

# Speed floor applied INSIDE the flow-coherence gate, as a multiple of the
# pre-gate noise sigma. This is a SECOND velocity selection, separate from
# DICOM_VESSEL_THRESHOLD_CMS above, and it is easy to miss: setting the
# vessel threshold to 0 disables the gate the console reports as "no
# velocity gate" while this floor keeps filtering silently.
#
# On the ideal phantom at 1.0 sigma it removed 124,489 voxels, 48.2% of the
# coherence-gated set — more than the vessel threshold it sits beneath —
# and 34% of those were above the rectified-noise level, i.e. genuine slow
# flow. That is exactly the population the in-vivo chapter argues a speed
# cut must not discard.
#
# It exists for a real reason: without it the noise estimator
# sigma = median(|v|)/1.5382 can be poisoned by a not-moving-at-all
# population, and on one earlier dataset it collapsed to an implied SNR of
# 3666. Measured on Series 11, though, sigma without the floor is 3.48 cm/s
# (implied SNR 43) against 5.14 cm/s (SNR 29) with it — physical either
# way. So the failure is dataset-specific, not general, and the floor
# should be a choice rather than a hidden constant.
#
# 0.0 disables it. Check the reported sigma after a run with it off: if the
# implied SNR is absurd, the estimator has been poisoned and the floor
# belongs back on, reported as a stated limitation.
DICOM_COHERENCE_SPEED_FLOOR_SIGMA: float = 1.0

# Lower bound on the flow threshold, as a FRACTION OF VENC. This stops the
# threshold collapsing to zero on very high-SNR data. It is deliberately
# relative, not an absolute cm/s value: an absolute floor tuned for cerebral
# arteries (~15 cm/s) would hide the flow completely in a slow pipe phantom,
# where VENC and the true velocities are an order of magnitude lower.
DICOM_VESSEL_MIN_VENC_FRAC: float           = 0.05

# Magnitude signal-mask threshold, as a fraction of the magnitude maximum.
# 0.10 is permissive: on a head it admits scalp, orbit and skull-base voxels
# whose SNR is 2-3, where the velocity noise is 14-20 cm/s. Raising it to
# 0.15-0.25 removes most of that at the cost of a few dim vessel voxels.
DICOM_MAGNITUDE_MASK_FRAC:  float           = 0.10

# When the signal mask lets BACKGROUND in (random-phase air inside the mask), every
# statistic measures air and the input checks abort the run with a misleading
# "storage order" / "magnitude data" error. With this on, the toolbox detects that
# from the data (the excess above VENC sits in the low-magnitude voxels only),
# raises the threshold to halfway between the background level and the
# object/background split, and CONTINUES - loudly, and recording what it did in
# run_config.json ("mask_auto_recovery"). It acts only when that diagnosis is
# positive, so a run whose mask is fine is untouched, and never when the fraction
# was set explicitly (--mask-frac): an explicit choice is respected and the run
# stops with the diagnosis instead. The recovered threshold is automatic, not
# tuned - confirm it with a threshold sweep before quoting numbers.
DICOM_AUTO_MASK_RECOVERY:   bool            = True

# Pipe preset only, and only when the DICOM header gives no velocity-encoding
# order (direction_source == 'assumed'): if the mask is a straight tube along one
# patient axis but the flow sits in another velocity component, the frame->axis
# assignment is wrong by construction (flow runs along the tube). Relabel the flow
# component to the tube axis (swap those two components), loudly, and record it.
# This fixes only the FLOW frame; the other two keep acquisition order, and the
# sign of the flow is not determined. --direction-order overrides it.
DICOM_AUTO_AXIS_ASSIGN:     bool            = True

# Pipe preset only: a threshold mask keeps EVERY bright voxel, including small
# objects beside the tube (fittings, holders, artefacts) whose phase is random.
# Their voxels read |v| ~ VENC, so they inflate the correction statistics,
# create spikes in the per-slice panel, and are counted as extra "channels" in the
# flux test. The premise of the pipe preset is ONE straight tube, so the analysis
# set is restricted to the largest connected object. Applied after the background
# noise reference is taken, so the noise estimate is unaffected. Set False to keep
# every bright voxel.
DICOM_PIPE_MAIN_OBJECT_ONLY: bool           = True
DICOM_MASK_FRAC_EXPLICIT:   bool            = False    # set by --mask-frac

# PC-MRA gate for the fig05 velocity panels, as a fraction of the 99.9th
# percentile of the PC-MRA (normalised magnitude x time-averaged speed) within
# the signal mask. DEFAULT OFF.
#
# HONEST NOTE ON WHAT THIS DOES AND DOES NOT FIX. It was added on the theory
# that time-averaging suppresses noise. It largely does not: |v| is positive-
# definite, so averaging shrinks the noise TAIL (p99.9 falls ~2x) but NOT its
# mean, which stays near 1.6*sigma. Tested against the case that actually
# spoils these panels — voxels with HIGH magnitude but corrupted phase (fat
# shift, motion, pulsation ghosting, orbit) — the gate rejected none of them,
# because such voxels are bright and fast and therefore have a high PC-MRA too.
# It also added nothing over the SNR-adaptive floor on ordinary low-SNR noise,
# which that floor already handles. Left here as something to try (set ~0.08),
# not as a claimed fix. The standard remedy is an anatomical brain mask —
# excluding everything outside the skull rather than separating it
# statistically.
DICOM_FIG05_PCMRA_GATE:     float           = 0.0

# Anatomical brain mask for IN VIVO head data. Scalp fat, orbits and skull-base
# tissue are bright on magnitude but their phase is corrupted by fat shift,
# motion and pulsation ghosting, so they survive every statistical threshold —
# they are bright AND fast. The only reliable remedy is to exclude them
# anatomically. The mask is built by eroding the magnitude mask (which breaks
# the thin scalp shell while the solid brain survives), keeping the largest
# connected component, then dilating back and filling holes. The dark skull
# separates scalp from brain on MR, which is what makes this work.
#
# SET FALSE FOR PHANTOM / PIPE DATA — there is no brain to find, and the
# erosion would eat a thin pipe lumen.
# Usability limit on the per-voxel velocity noise, as a fraction of VENC.
# A voxel whose velocity uncertainty is a large fraction of the whole encoding
# range carries no velocity information — it is a signal-dropout region
# (frontal sinus, petrous bone, mastoid), not a measurement. Such voxels are
# EXCLUDED from the mask, the statistics and the figure. Capping their
# threshold instead is worse than useless: it lowers the bar precisely where
# the data is least trustworthy, which is what put >VENC speckle in fig05.
DICOM_MAX_SIGMA_V_FRAC:     float           = 0.20

# Minimum connected-component size, in voxels, for a region to be DISPLAYED in
# the fig05 velocity panels. 0 disables.
#
# A maximum-intensity projection lights an entire column if ANY voxel in it is
# bad, so a sparse artifact population is amplified enormously: at 1% corrupted
# voxels a 52-slice projection paints 41% of the panel. Those artifacts (fat
# chemical shift, pulsation ghosting, susceptibility near sinuses) are isolated
# single voxels, whereas a vessel is a contiguous tube of hundreds. Discarding
# small components before projecting removes the speckle and keeps the vessels.
# DISPLAY ONLY — it never touches the correction or any reported statistic.
DICOM_FIG05_MIN_BLOB:       int             = 8

DICOM_BRAIN_MASK:           bool            = False
DICOM_BRAIN_MASK_ERODE:     int             = 4

# Restrict the fig05 DIFFERENCE panel to flow voxels.
#
# BUG FIX (pre-existing). This name was previously created ONLY by
# _apply_dataset_preset() through globals()[k] = v. That function returns
# early for DATASET_PRESET = "manual", so with the documented "manual"
# setting the name never existed and _dicom_quick_plot() raised
#     NameError: name 'DICOM_FIG05_DIFF_ON_FLOW_ONLY' is not defined
# at figure time — after the whole correction had already run. Every preset
# sets it True, so a module-level default of True changes nothing for the
# preset paths and simply makes "manual" work.
DICOM_FIG05_DIFF_ON_FLOW_ONLY: bool         = True

# Save the measured + corrected volumes as dicom_corrected.npz.
#
# This is a convenience artefact for re-analysis, not a result. It holds two
# float32 copies of the full 5-D velocity array, so for a 3x224x224x88x11
# acquisition that is ~1.2 GiB in memory BEFORE compression — enough to fail
# on a laptop, and the failure used to abort fig05 and the JSON summary with
# it. The corrected field is exported to VTK regardless, which is what
# ParaView needs.
# Coronal plane used for the Lambda-matrix and VENC(r) figures, in mm from
# isocentre. NOT zero by default, and the reason is physical rather than
# cosmetic.
#
# y = 0 is a SYMMETRY PLANE of the gradient system: four of the nine cross
# terms vanish there identically. Measured on the SC72 coil, max |Lambda_ij|
# over the slice:
#
#     y [mm]     L_xy      L_yx      L_yz      L_zy
#      +0.0    0.00000   0.00000   0.00000   0.00000
#      +9.2    0.016     0.022     0.024     0.042
#     +55.0    0.099     0.132     0.147     0.248
#    +100.8    0.181     0.230     0.270     0.446
#
# So a figure drawn at exactly y = 0 shows four blank panels and understates
# the very effect the generalised reconstruction exists to correct: it is the
# OFF-DIAGONAL terms that rotate the encoding direction, and a reader seeing
# them empty would conclude the coupling is negligible. (Markl et al. 2003
# Fig. 1 shows y = 0 and an off-centre plane side by side for this reason.)
#
# Only the display is affected — every metric uses the whole volume.
# Set to 0.0 to reproduce the symmetry plane deliberately.
FIG_CORONAL_SLICE_MM:       float           = 55.0

DICOM_SAVE_NPZ:             bool            = True
DICOM_NPZ_MAX_BYTES:        int             = 3 * (2 ** 30)   # 3 GiB

# Thickness of the fig05 image panels, in slices EITHER SIDE of the display
# slice, rendered as a maximum-intensity projection. The Circle of Willis is a
# 3-D ring spanning ~10-15 mm in z, so a single sub-millimetre slice can never
# contain it. 8 slices either side at 0.64 mm covers ~11 mm. Set 0 for a
# single slice (the classic single-plane panel). Thicker slabs assemble more
# of the ring but raise the noise floor (a MIP reports the largest of N noise
# samples), which the display threshold compensates for automatically.
DICOM_FIG05_SLAB:           int             = 8   # -1 = project the WHOLE volume

# ── VTK export for ParaView ─────────────────────────────────────────────────
# Which cardiac phase(s) write_velocity_vtk() writes.
#   "peak"  (default) — one file at peak systole (the phase with the highest
#                       mean speed inside the object). ~225 MB at 260x320x52.
#   "all"             — every phase, as velocity_field_t00.vtk … _tNN.vtk.
#                       ParaView auto-detects the numbered series as ONE
#                       dataset and gives you the VCR play button, which is
#                       what you want to animate the cardiac cycle.
#   <int>             — force a single phase index.
#
# "all" is large: 17 phases x 4.3M voxels. DICOM_VTK_LEAN below keeps it to
# ~1.8 GB instead of ~3.8 GB by not repeating, in all 17 files, data that
# either does not change with time or that ParaView can derive itself.
DICOM_VTK_PHASES = "peak"          # "peak" | "all" | int

# In time-series mode, write only the two velocity VECTORS per phase and put
# the static anatomy into a single companion file. The dropped arrays are not
# lost: ParaView's colour-by menu offers "Magnitude" for any vector array
# (replacing speed_measured / speed_corrected), and the Calculator filter
# gives the correction vector as velocity_corrected - velocity_measured.
# The anatomy magnitude is time-AVERAGED upstream, so it is byte-identical in
# every phase — repeating it 17 times costs ~290 MB for no information.
# Ignored for single-phase exports, which always write the full set.
DICOM_VTK_LEAN:             bool            = True


# ── Apply the preset ─────────────────────────────────────────────────────────
# Written as a function so the choice can be changed per run from the GUI
# without editing this file. Selecting one preset never alters the other: the
# in-vivo settings are restored in full whenever "in_vivo_head" is chosen, so a
# phantom run cannot leave anything behind that affects the next head run.
DATASET_PRESETS = {
    "in_vivo_head": dict(          # brain 4D flow (e.g. Circle of Willis)
        # OFF for brain. Parenchyma is already handled by the magnitude and
        # brain masks, and a coherence cut here would discard genuine slow or
        # pulsatile vessels — cerebral veins in particular.
        DICOM_FIG05_DIFF_ON_FLOW_ONLY = True,   # parenchyma is noise; the CoW figure already worked this way
        DICOM_FLOW_COHERENCE_MIN   = None,
        DICOM_BRAIN_MASK           = True,   # remove scalp / orbit / skull-base
        DICOM_FIG05_SLAB           = -1,     # full-volume MIP: whole vessel tree
        DICOM_MAGNITUDE_MASK_FRAC  = 0.10,
        DICOM_VESSEL_SIGMA_MULT    = 4.0,
        # Reset per dataset so a value tuned for one never silently applies to
        # the other. A colour-scale cap chosen for a phantom would compress a
        # head study; a direction permutation for one acquisition would corrupt
        # another.
        DICOM_FIG05_VMAX_CMS       = None,   # automatic
        DICOM_DIRECTION_ORDER      = None,   # auto-detect from the header
        # No velocity gate: include every magnitude-masked voxel so that slow
        # vessels in the Circle of Willis are not discarded. Noise rejection
        # is left to the magnitude + brain masks. The run prints the
        # noise-consistent fraction of this set — quote it alongside the
        # correction numbers (see DICOM_VESSEL_THRESHOLD_CMS).
        DICOM_VESSEL_THRESHOLD_CMS = 0.0,
        # No coherence gate on brain data, so this never fires; 0.0 records
        # the intent that no velocity selection is applied anywhere.
        DICOM_COHERENCE_SPEED_FLOOR_SIGMA = 0.0,
        DICOM_COHERENT_SUBSET_MIN  = 0.5,
    ),
    "ideal_phantom": dict(         # ideal / cardiac-cycle phantom
        # ON. The ideal rig carries a large STATIC WATER SPHERE beside the flow
        # channel. Both are bright water, so the magnitude mask keeps both and
        # a speed cut cannot separate them: the vessel test is applied to
        # mean_t(|v|), the mean of the MAGNITUDE, which rectifies noise to
        # 1.6*sigma (~5 cm/s here) — a positive number no averaging removes.
        # Measured on the Series 11-9 data, the sphere put 189,309 voxels of
        # stationary water into the "vessel" mask, 18.6% of all signal.
        #
        # 0.60 was chosen against the physics, not tuned: static fluid sits at
        # 1/sqrt(Nt) = 0.30 for Nt = 11, real flow near 1.0. On a synthetic
        # replica of this phantom 0.60 removed 99% of the sphere and kept 100%
        # of the flow channel.
        DICOM_FIG05_DIFF_ON_FLOW_ONLY = True,   # the static water sphere dwarfs the flow channel
        DICOM_FLOW_COHERENCE_MIN   = 0.60,
        DICOM_BRAIN_MASK           = False,  # no brain; erosion would eat the lumen
        DICOM_FIG05_SLAB           = 0,      # single plane through the phantom
        DICOM_MAGNITUDE_MASK_FRAC  = 0.10,
        DICOM_VESSEL_SIGMA_MULT    = 3.0,    # phantoms are high-SNR; 4σ is strict
        DICOM_FIG05_VMAX_CMS       = None,
        DICOM_DIRECTION_ORDER      = None,
        # NO VELOCITY GATE — 0.0, matching in_vivo_head.
        #
        # Previously None, which derived a 3σ cut and gave 7.5 cm/s on the
        # Series 11 data. That excluded 4,238 voxels of the coherent set
        # (19%) whose speeds ran 3.19–7.50 cm/s — ALL of them above the
        # 3.81 cm/s rectified-noise floor, so genuine slow flow, not noise.
        #
        # The in-vivo chapter argues that a speed threshold cannot separate
        # slow flow from rectified noise and therefore uses no velocity gate.
        # Applying one here would make the phantom chapter contradict that
        # argument, so the gate is off in both. Selection is left to the
        # magnitude mask and the flow-coherence cut, neither of which is a
        # measure of how fast the fluid moves.
        #
        # WHAT THIS COSTS, and it must be reported rather than hidden: the
        # static water sphere is 125,626 voxels and 51.7% of it passes
        # coherence ≥ 0.60 on the real data (the 99% rejection quoted above
        # was measured on a SYNTHETIC replica). With no speed gate nothing
        # else stops those voxels, and 32% of the analysis set then sits at
        # or below the noise floor. Quote that fraction beside the correction
        # numbers, exactly as the in-vivo run quotes its own 66%.
        #
        # The headline number is unaffected: mean relative correction is
        # 2.64% with the gate and 2.65% without.
        #
        # The coherence cut is deliberately NOT raised to compensate. 0.60
        # follows from the physics — static fluid sits at 1/sqrt(Nt) = 0.30
        # for Nt = 11, real flow near 1.0 — and moving it to offset the
        # removal of the speed gate would be tuning the selection to the
        # dataset rather than to the measurement.
        DICOM_VESSEL_THRESHOLD_CMS = 0.0,
        # And no speed floor inside the coherence gate either — see
        # DICOM_COHERENCE_SPEED_FLOOR_SIGMA. With the vessel threshold at 0
        # that floor would otherwise become the ONLY velocity selection, and
        # on this phantom it was removing 48% of the coherent set, a third of
        # it genuine slow flow. Selection is now coherence alone, matching
        # in_vivo_head, and the noise-consistent fraction is reported instead.
        DICOM_COHERENCE_SPEED_FLOOR_SIGMA = 0.0,
        DICOM_COHERENT_SUBSET_MIN  = None,
    ),
    "pipe_flow": dict(             # steady / pulsatile pipe rig
        # ON. Pipe rigs usually sit in a water bath or beside a reference
        # bottle, which is the same static-fluid problem as the ideal phantom.
        DICOM_FIG05_DIFF_ON_FLOW_ONLY = True,   # water bath / reference bottle beside the pipe
        DICOM_FLOW_COHERENCE_MIN   = 0.60,
        DICOM_BRAIN_MASK           = False,
        DICOM_FIG05_SLAB           = 0,
        # A percent-of-max threshold only works when the background sits
        # near zero. On the Skyra pipe-rig data the air pedestal is ~63 and
        # the peak only 281, so 10% of max (28) lies BELOW the background
        # and the mask keeps 93% of the volume — the pipeline then measures
        # air, whose random phase gives |v| ~ VENC and trips the strict
        # input check. Air p99 = 74, air max = 83, so 0.30 (threshold 84)
        # is the first fraction that clears the noise and yields a single
        # coherent object. Re-check with check_magnitude_mask.py if the
        # magnitude scaling of a new dataset differs.
        # None = chosen from the data, per series: halfway between the background
        # level and the object/background split (Otsu on the non-zero voxels). A fixed
        # fraction of the maximum does not transfer between acquisitions: 0.30 was
        # tuned on one series and clips the dimmer end of the tube on another (on the
        # 1 mm series it cut the lumen area CV to 23% and Q by 9%), while the
        # halfway rule sat mid-plateau (0.18 of max, area CV 1.3%). Set a number
        # here, or --mask-frac F, to override.
        DICOM_MAGNITUDE_MASK_FRAC  = None,
        DICOM_VESSEL_SIGMA_MULT    = 3.0,
        DICOM_FIG05_VMAX_CMS       = None,
        DICOM_DIRECTION_ORDER      = None,   # set per run: --direction-order (see CLI help)
        # NO VELOCITY GATE — 0.0, the rule the in-vivo and ideal-phantom presets
        # already follow (a speed threshold cannot separate slow flow from
        # rectified noise, so selection is left to the magnitude mask and, where
        # the data are time-resolved, the flow-coherence cut). This was None
        # (automatic 3 sigma). A steady pipe series has ONE time point, so the
        # coherence gate cannot run (it needs >= 4) and the analysis set is the
        # magnitude mask alone; the noise-consistent fraction is reported.
        # Set it back to None to restore the automatic 3-sigma threshold.
        DICOM_VESSEL_THRESHOLD_CMS = 0.0,
        # Left at the historical 1 sigma. A pipe rig sits in a water bath,
        # the same static-fluid problem as the ideal phantom, but the pipe
        # data has not been examined yet - so this keeps the previous
        # behaviour rather than assuming the phantom's answer transfers.
        # Revisit once the pipe geometry and noise are measured.
        DICOM_COHERENCE_SPEED_FLOOR_SIGMA = 1.0,
        DICOM_COHERENT_SUBSET_MIN  = None,
    ),
}
# "phantom" kept as an alias so an older DATASET_PRESET string still resolves.
DATASET_PRESETS["phantom"] = dict(DATASET_PRESETS["ideal_phantom"])

# Every preset MUST define exactly the same set of keys. A key present in one
# preset but missing from another would not be reset when switching, so a value
# tuned for the phantom would silently survive into a Circle-of-Willis run (or
# the reverse) — the precise failure this preset mechanism exists to prevent.
# Checked at import so a future edit cannot reintroduce it unnoticed.
_preset_keysets = {n: frozenset(d) for n, d in DATASET_PRESETS.items()}
if len(set(_preset_keysets.values())) != 1:
    _union = frozenset().union(*_preset_keysets.values())
    _detail = "; ".join(
        f"{n} missing {sorted(_union - k)}"
        for n, k in _preset_keysets.items() if k != _union)
    raise RuntimeError(
        "DATASET_PRESETS are not symmetric — every preset must define the "
        f"same keys or settings leak between datasets. {_detail}")


def _apply_dataset_preset(preset: str, announce: bool = False) -> None:
    """
    Set the coupled DICOM_* options from a named preset.

    The names are rebound through globals()[k] = v below, so the module-level
    DICOM_* settings are updated in place.
    """
    # Record the preset that is actually IN FORCE. Until this line the module-level
    # DATASET_PRESET stayed at the file default ("ideal_phantom") whatever the
    # dialog or --preset chose, so anything keyed on it saw the wrong dataset type:
    # the anatomical sign check (which runs only for brain data) read
    # "ideal_phantom" on every GUI/CLI run and was therefore skipped for
    # Circle-of-Willis runs too, and its message named the wrong preset.
    global DATASET_PRESET
    if preset in DATASET_PRESETS or preset == "manual":
        DATASET_PRESET = preset
    if preset == "manual":
        if announce:
            print("  Dataset preset: manual — using the DICOM_* settings "
                  "exactly as written in the file")
        return
    if preset not in DATASET_PRESETS:
        raise ValueError(
            "DATASET_PRESET must be one of "
            f"{sorted(DATASET_PRESETS) + ['manual']}, got {preset!r}")
    for k, v in DATASET_PRESETS[preset].items():
        globals()[k] = v
    # DICOM_DIRECTION_PERMUTE / _SIGNS encode a component mapping verified on ONE
    # acquisition. They default to None now; this reset is kept as a second
    # guard for the pipe preset, whose axes are assigned from the geometry.
    # main() applies --direction-permute / --direction-signs AFTER the preset,
    # so an explicit command-line mapping still reaches a pipe run.
    if preset == "pipe_flow":
        globals()["DICOM_DIRECTION_PERMUTE"] = None
        globals()["DICOM_DIRECTION_SIGNS"] = None
    if announce:
        print(f"  Dataset preset: {preset}  "
              f"(brain mask {'ON' if DICOM_BRAIN_MASK else 'OFF'}, "
              f"fig05 slab {DICOM_FIG05_SLAB}, "
              f"vessel cut {DICOM_VESSEL_SIGMA_MULT:g}σ)")


_apply_dataset_preset(DATASET_PRESET)


# ═════════════════════════════════════════════════════════════════════════════
# MATCHING THE SIMULATION TO THE ACQUISITION
# ═════════════════════════════════════════════════════════════════════════════
#
# The synthetic experiment in fig02/fig04 is only informative if it is run at
# the VENC, phase count and noise level of the data it sits beside: the
# percentage improvement there is dominated by the assumed SNR, because after
# correction the residual is essentially pure noise.
#
# NO DATASET NAMES APPEAR HERE. An earlier version carried a lookup table
# keyed by acquisition ("cow", "phantom_div11_9", ...), which meant the file
# had to be edited for every new dataset — the opposite of a general toolbox.
# The values are taken from the acquisition itself instead:
#
#   VENC  - the value entered in the dialog or passed with --venc
#   Nt    - the number of cardiac phases found in the DICOM series
#   snr_db- from the velocity noise floor the DICOM stage measures, via
#             SNR_dB = 20*log10( sqrt(2)*VENC / (pi*sigma_v) )
#
# Only VENC is known before the simulation runs, so VENC and Nt are matched
# automatically and the SNR is checked afterwards: if the simulation was run
# at a level the data does not support, the DICOM stage prints the value to
# use. Set SIM_MATCH_ACQUISITION = False to keep SimulationConfig as written,
# or pass --snr-db / --sim-venc / --sim-nt to fix any of them by hand.
SIM_MATCH_ACQUISITION: bool = True

# Set by --grid-n; None means "use SimulationConfig.grid_N".
#
# MUST BE DECLARED HERE. It is only assigned inside the command-line branch
# of main(), so without a module-level binding the GUI path — which never
# enters that branch — reached `if _CLI_GRID_N:` with the name unbound and
# died with NameError before Stage 2, after the coefficient file had already
# parsed. Anything a `global` statement assigns conditionally needs a
# definition at import time.
_CLI_GRID_N: Optional[int] = None

# Command-line overrides; None means "use SimulationConfig / auto-match".
SIM_SNR_DB_OVERRIDE: Optional[float] = None
SIM_VENC_OVERRIDE:   Optional[float] = None
SIM_NT_OVERRIDE:     Optional[int]   = None


def _apply_sim_overrides(cfg_obj, acq_venc_ms: Optional[float] = None,
                         acq_nt: Optional[int] = None) -> None:
    """Match the simulation to the acquisition, then apply any manual override."""
    _msg = []
    if SIM_MATCH_ACQUISITION and acq_venc_ms:
        if abs(float(acq_venc_ms) - cfg_obj.VENC) > 1e-6:
            _msg.append(f"VENC {cfg_obj.VENC:.2f} -> {float(acq_venc_ms):.2f} m/s")
            cfg_obj.VENC = float(acq_venc_ms)
    if SIM_MATCH_ACQUISITION and acq_nt and int(acq_nt) > 1:
        if int(acq_nt) != cfg_obj.Nt:
            _msg.append(f"Nt {cfg_obj.Nt} -> {int(acq_nt)}")
            cfg_obj.Nt = int(acq_nt)
    if SIM_VENC_OVERRIDE is not None:
        cfg_obj.VENC = float(SIM_VENC_OVERRIDE); _msg.append("VENC set by --sim-venc")
    if SIM_NT_OVERRIDE is not None:
        cfg_obj.Nt = int(SIM_NT_OVERRIDE);       _msg.append("Nt set by --sim-nt")
    if SIM_SNR_DB_OVERRIDE is not None:
        cfg_obj.snr_db = float(SIM_SNR_DB_OVERRIDE)
        _msg.append(f"snr_db set by --snr-db = {cfg_obj.snr_db:.0f}")
    if _msg:
        print("  [sim] Matched to the acquisition: " + "; ".join(_msg))
    else:
        print("  [sim] SimulationConfig used as written "
              "(no acquisition values available yet).")


# ─────────────────────────────────────────────────────────────────────────────
# Figure output selection + sequential renumbering.
# A maximum of five figures is produced, including
# the DICOM figure. Only the figures listed here are written to disk; all
# others are still computed internally but not saved. On output they are
# renamed to sequential fig01..fig05 in the narrative order below.
# original 2 → fig01 problem: voxel-wise Λ(r) distortion/correction operator
# original 7 → fig02 error caused: % velocity error before vs after
# original 17 → fig03 correction concept: spatially varying VENC (GNL signature)
# original 12 → fig04 quantitative validation: RMSE / MAE before vs after
# original 19 → fig05 real-world application: acquired DICOM correction (required)
# FIG_RENUMBER is the single source of truth: key = original figure number,
# value = new sequential position. To restore all 19 original figures, set
# FIG_RENUMBER = {} and ENABLED_FIGS = None.
FIG_RENUMBER = {2: 1, 7: 2, 17: 3, 12: 4, 19: 5}
ENABLED_FIGS = set(FIG_RENUMBER)   # figures kept, keyed by ORIGINAL number


def isolate_flow_lumens(signal_mask, flow_mask,
                        min_flow_voxels: int = 200, max_lumens: int = 6,
                        min_flow_fraction: float = 0.25):
    """
    Split a phantom's signal mask into separate flow CHANNELS.

    WHY THIS IS NEEDED. Volumetric flow rate Q is defined for a single channel
    carrying flow in one direction. Summed over a mask that spans several
    lumens, the contributions cancel or reinforce arbitrarily: on the cardiac
    phantom that produced Q values whose plane-to-plane spread was 423%, and
    the code correctly refused to quote them. Labelling the mask into connected
    components and treating each channel separately is what makes Q meaningful.

    WHY THE MASK COMES FROM THE MAGNITUDE, NOT THE SPEED. The vessel mask is a
    speed threshold (~3 sigma), which keeps the fast core of a lumen and drops
    the slow annulus near the wall — on the cardiac phantom roughly half the
    lumen area. Integrating Q over the fast core alone would understate the
    flow rate systematically, and would do so MORE where the profile is
    blunter. So the channel is taken from the magnitude signal mask, with
    holes filled, and the flow mask is used only to decide WHICH components
    are channels rather than static fluid.

    A COMPONENT IS ONLY A CHANNEL IF IT IS MOSTLY FLUID IN MOTION. On in-vivo
    head data the magnitude signal mask is a single connected blob — the whole
    head — and it does contain thousands of flow voxels, so a count threshold
    alone accepts it as "channel 1". That happened: the Circle-of-Willis run
    returned one component of 1,377,181 voxels of which 16,836 (1.2%) were
    flowing, and the flux test then integrated v_y over whole-brain
    cross-sections. Q rose from 1 to 2094 ml/min along y purely because each
    plane cuts a different amount of anatomy, giving CV = 77% that measures
    neither the correction nor any error.

    `min_flow_fraction` rejects that: a genuine lumen is predominantly moving
    fluid. A phantom tube scores ~0.5 (the slow wall annulus falls below the
    speed threshold), a head scores ~0.01.

    Returns a list of dicts, largest channel first:
        {"mask": bool (X,Y,Z), "n_voxels", "n_flow_voxels", "flow_fraction",
         "centroid_vox"}
    """
    try:
        from scipy import ndimage
    except ImportError:
        print("  [lumen] scipy unavailable — per-channel flow rate skipped.")
        return []

    sig = np.asarray(signal_mask, dtype=bool)
    flo = np.asarray(flow_mask, dtype=bool)
    lab, n_lab = ndimage.label(sig)
    if n_lab == 0:
        return []

    # How much moving fluid does each connected component hold? A static water
    # bath is a large component with almost no flow voxels and is rejected here.
    flow_per_label = np.bincount(lab[flo].ravel(), minlength=n_lab + 1)
    out = []
    for i in range(1, n_lab + 1):
        n_flow = int(flow_per_label[i]) if i < flow_per_label.size else 0
        if n_flow < min_flow_voxels:
            continue
        comp = ndimage.binary_fill_holes(lab == i)
        n_vox = int(comp.sum())
        frac = n_flow / max(n_vox, 1)
        if frac < min_flow_fraction:
            print(f"  [lumen] component of {n_vox:,} voxels rejected: only "
                  f"{100*frac:.1f}% of it is flowing "
                  f"(needs >= {100*min_flow_fraction:.0f}%). This is bulk "
                  "anatomy or a static bath, not a flow channel.")
            continue
        idx = np.argwhere(comp)
        out.append({"mask": comp,
                    "n_voxels": n_vox,
                    "n_flow_voxels": n_flow,
                    "flow_fraction": float(frac),
                    "centroid_vox": tuple(idx.mean(axis=0).round(1))})
    out.sort(key=lambda d: -d["n_flow_voxels"])
    return out[:max_lumens]


def stable_area_span(lumen, axis: int, max_cv_pct: float = 40.0,
                     min_planes: int = 8):
    """
    Longest contiguous run of planes along `axis` whose cross-section is steady.

    WHY THIS EXISTS. Mass conservation fixes Q along a channel with ONE inlet
    and ONE outlet. A real rig is rarely only that: on the ideal phantom the
    largest connected flow component is a tube JOINED TO A WIDE CHAMBER, and
    the two are physically continuous, so component labelling returns them as
    one object. Slicing it along the flow axis gives areas of roughly 30 voxels
    through the tube and 148-289 through the chamber, a cross-sectional
    variation of 147% — and flux_conservation correctly refuses that, because
    the area changes for CONSTRUCTION reasons, not measurement error.

    Cutting at the junction recovers a usable geometry: on that phantom the
    tube section alone runs 90 planes at 16% area variation, well inside the
    60% the test allows.

    The cut is purely GEOMETRIC — it reads the cross-sectional area profile and
    nothing else. No velocity, no Lambda, no corrected field enters the choice,
    so the segmentation cannot bias the validation it feeds. That matters: this
    is the only check in the pipeline that does not depend on Lambda, and it
    would be worthless if the region it runs on were selected using Lambda.

    The existing end-plane trim in tube_axis_and_planes() is not sufficient
    here. It keeps planes with area >= 0.5*median, which removes the tapering
    ends but KEEPS a chamber, whose area is far ABOVE the median rather than
    below it.

    Returns (keep_plane_indices, cv_pct) or (None, None) if no run qualifies.
    """
    lum = np.asarray(lumen, bool)
    n = lum.shape[axis]
    area = np.array([int(np.take(lum, k, axis=axis).sum())
                     for k in range(n)], float)
    occ = np.flatnonzero(area > 0)
    if occ.size < min_planes:
        return None, None

    def cv_of(i, j):
        seg = area[occ[i]:occ[j] + 1]
        seg = seg[seg > 0]
        if seg.size < min_planes:
            return None, 0
        return float(100.0 * seg.std() / max(seg.mean(), 1e-9)), seg.size

    best = None                       # (length, i, j, cv)
    for i in range(occ.size - min_planes + 1):
        # longest run starting at i: walk j down until the spread is acceptable
        for j in range(occ.size - 1, i + min_planes - 2, -1):
            cv, cnt = cv_of(i, j)
            if cv is None:
                continue
            if cv <= max_cv_pct:
                if best is None or (j - i) > (best[0]):
                    best = (j - i, i, j, cv)
                break                 # nothing longer can start at this i
    if best is None:
        return None, None
    _, i, j, cv = best
    keep = occ[(occ >= occ[i]) & (occ <= occ[j])]
    keep = keep[area[keep] > 0]
    return keep, cv


def tube_axis_and_planes(lumen, min_area_frac: float = 0.5):
    """
    Pick the grid axis a tube actually runs along, and drop its end planes.

    WHY BOTH STEPS ARE NEEDED. Q must be integrated over planes NORMAL to the
    channel. Choosing the axis from the dominant velocity component looked
    reasonable and was wrong for Series 11-9: it gave cross-sections whose
    area varied by 62% over 34 planes with one lumen per plane — not a
    branching network, just oblique cuts through a tube, plus the tapering
    planes where the tube leaves the field of view.

    A tube has a constant cross-section normal to its own axis, so the axis is
    identified by measuring the area profile along all three and taking the
    steadiest. End planes, where the lumen is only partly inside the volume,
    are then trimmed: they carry a genuinely smaller cross-section, so the
    lower Q there is geometry rather than measurement error and including it
    would inflate the spread the flux test is meant to measure.

    Returns (axis, keep_planes, area_cv_pct_after_trim).
    """
    best = (None, None, np.inf)
    for ax in range(3):
        n = lumen.shape[ax]
        area = np.array([int(np.take(lumen, k, axis=ax).sum())
                         for k in range(n)], float)
        occ = np.flatnonzero(area > 0)
        if occ.size < 5:
            continue
        med = float(np.median(area[occ]))
        keep = occ[area[occ] >= min_area_frac * med]
        if keep.size < 5:
            continue
        cv = float(100.0 * np.std(area[keep]) / max(np.mean(area[keep]), 1e-9))
        # Prefer the steadiest cross-section; break ties toward more planes,
        # since a longer run of the tube gives a better-determined spread.
        if cv < best[2] - 1e-9:
            best = (ax, keep, cv)
    return best


def flux_conservation(v_meas3, v_corr3, lumen, spacing_m, axis: int):
    """
    Q through every plane of one channel, before and after correction.

    THIS IS THE ONLY CHECK IN THE PIPELINE THAT NEEDS NO EXTERNAL REFERENCE
    AND DOES NOT DEPEND ON LAMBDA. For an incompressible fluid in a rigid
    channel at steady state, conservation of mass requires Q to be IDENTICAL
    through every cross-section. Any drift along the channel is measurement
    error. Gradient nonlinearity varies with position, so it makes Q drift
    with distance from isocentre; a correction that models the field
    correctly should flatten that drift.

    The simulation cannot test this (it shares Lambda with the correction) and
    the Lambda-map correlations cannot either (they are consistency checks).
    The plane-to-plane spread is a physical constraint imposed from outside
    the model, which is what makes it validation rather than verification.

    Returns a dict; `cv_*_pct` is the coefficient of variation of Q across
    planes, and a DROP from before to after is the result of interest.
    """
    q_b, planes = dicom_flow_rate(v_meas3, lumen, spacing_m, axis)
    q_a, _      = dicom_flow_rate(v_corr3, lumen, spacing_m, axis)

    # Drop the end planes where the tube is only partly inside the volume.
    # NOTE: tube_axis_and_planes() returns the trim for the axis IT would have
    # chosen. When the caller overrides that axis — which it does whenever the
    # geometric axis does not carry the flow — the trim below used to be
    # skipped entirely, leaving the untrimmed profile to fail the area test. So
    # recompute the trim for the axis actually in use.
    _ax_sel, _keep, _cv_trim = tube_axis_and_planes(lumen)
    if _ax_sel != axis:
        _n_ax = np.asarray(lumen, bool).shape[axis]
        _ar = np.array([int(np.take(np.asarray(lumen, bool), k, axis=axis).sum())
                        for k in range(_n_ax)], float)
        _oc = np.flatnonzero(_ar > 0)
        if _oc.size >= 5:
            _med = float(np.median(_ar[_oc]))
            _keep = _oc[_ar[_oc] >= 0.5 * _med]
        else:
            _keep = None
    if _keep is not None and len(_keep):
        _sel = np.isin(np.asarray(planes), np.asarray(_keep))
        if _sel.sum() >= 3 and _sel.sum() < len(planes):
            print(f"           trimmed {len(planes) - int(_sel.sum())} end "
                  f"plane(s) where the lumen leaves the volume")
        if _sel.sum() >= 3:
            q_b, q_a = q_b[_sel], q_a[_sel]
            planes = list(np.asarray(planes)[_sel])

    # Cut a channel that is a tube JOINED TO A CHAMBER down to the tube. See
    # stable_area_span(): the cut is geometric only, so it cannot bias the one
    # test here that is independent of Lambda.
    _span_note = None
    _kp2, _cv2 = stable_area_span(lumen, axis)
    if _kp2 is not None and len(_kp2):
        _sel2 = np.isin(np.asarray(planes), np.asarray(_kp2))
        if 3 <= int(_sel2.sum()) < len(planes):
            _span_note = {
                "planes_before": int(len(planes)),
                "planes_used": int(_sel2.sum()),
                "area_cv_pct_used": float(_cv2),
                "why": ("the channel is not a single tube over its whole "
                        "length; Q is reported over the longest run of planes "
                        "whose cross-section is steady. The cut uses the area "
                        "profile only — no velocity and no Lambda."),
            }
            print(f"           restricted to the steady-cross-section run: "
                  f"{int(_sel2.sum())} of {len(planes)} planes, area CV "
                  f"{_cv2:.0f}%")
            q_b, q_a = q_b[_sel2], q_a[_sel2]
            planes = list(np.asarray(planes)[_sel2])
    if q_b.size < 3:
        return {"usable": False,
                "why": f"only {int(q_b.size)} planes hold a usable cross-section"}

    # DOES THE FLOW ACTUALLY GO THROUGH THESE PLANES? An axis orthogonal to
    # the flow gives Q ~ 0 through every plane, hence a coefficient of
    # variation of ~0 — a perfectly "conserved" zero that would pass every
    # other check here and report a meaningless flow rate. Require the
    # through-plane component to carry a real share of the speed.
    try:
        _lu = np.asarray(lumen, bool)
        _sp = np.sqrt((np.asarray(v_meas3)[:, _lu] ** 2).sum(axis=0)).mean()
        _tp = np.abs(np.asarray(v_meas3)[axis][_lu]).mean()
        _share = float(_tp / max(_sp, 1e-12))
    except Exception:
        _share = 1.0
    if _share < 0.30:
        return {"usable": False,
                "why": (f"the flow is not along {'xyz'[axis]}: the "
                        f"through-plane component carries only "
                        f"{100*_share:.0f}% of the speed, so Q through these "
                        "planes is near zero and its spread is meaningless"),
                "through_plane_share": _share,
                "n_planes": int(q_b.size)}

    # IS THIS ONE TUBE, OR A NETWORK? Mass conservation fixes Q along a single
    # channel with one inlet and one outlet. Through a BRANCHING tree it does
    # not: each plane cuts a different number of vessels, so Q varies for
    # anatomical reasons and the spread means nothing. Count the 2-D connected
    # components in each cross-section — a tube gives 1, a vascular tree gives
    # several — and refuse to report a spread that cannot be interpreted.
    n_comp, area = [], []
    try:
        from scipy import ndimage as _ndi
        for k in planes:
            sl = [slice(None)] * 3
            sl[axis] = int(k)
            face = np.asarray(lumen)[tuple(sl)]
            area.append(int(face.sum()))
            n_comp.append(int(_ndi.label(face)[1]))
    except Exception:
        n_comp, area = [1], [1]
    med_comp = float(np.median(n_comp))
    area_cv = float(100.0 * np.std(area) / max(np.mean(area), 1e-9))
    if med_comp > 1.5 or area_cv > 60.0:
        return {"usable": False,
                "why": (f"not a single channel: cross-sections hold "
                        f"{med_comp:.0f} separate lumens on average and the "
                        f"area varies by {area_cv:.0f}% along the axis. Q is "
                        "defined for one tube with one inlet and one outlet; "
                        "through a branching network it changes for "
                        "anatomical reasons and the spread is not error."),
                "median_lumens_per_plane": med_comp,
                "area_cv_pct": area_cv,
                "n_planes": int(q_b.size)}

    # A channel whose Q changes sign across planes is not one channel with one
    # flow direction; the mean is then meaningless and so is the CV.
    sign_flip = bool(np.any(q_b > 0) and np.any(q_b < 0))
    cv = lambda q: float(100.0 * np.std(q) / max(abs(np.mean(q)), 1e-9))
    return {
        "usable": not sign_flip,
        "why": ("Q changes sign across planes — the mask still spans more than "
                "one flow direction" if sign_flip else "single channel, "
                "single direction"),
        "normal_axis": "xyz"[axis],
        "n_planes": int(q_b.size),
        "Q_before_ml_min": float(np.mean(q_b)),
        "Q_after_ml_min":  float(np.mean(q_a)),
        "Q_before_median_ml_min": float(np.median(q_b)),
        "Q_after_median_ml_min":  float(np.median(q_a)),
        "cv_before_pct": cv(q_b),
        "cv_after_pct":  cv(q_a),
        "cv_improvement_pct": float(100.0 * (cv(q_b) - cv(q_a))
                                    / max(cv(q_b), 1e-9)),
        "q_before_ml_min": [float(v) for v in q_b],
        "q_after_ml_min":  [float(v) for v in q_a],
        "plane_index":     [int(v) for v in planes],
        "segmentation":    _span_note,
    }


def array_axes_to_patient(iop, tol_deg: float = 5.0, return_signs: bool = False):
    """
    Which PATIENT axis (0 = x, 1 = y, 2 = z in LPS) does each ARRAY axis run along?

    The DICOM loader stores volumes as (columns, rows, slices) = array axes
    (0, 1, 2), which run along  IOP[0:3],  IOP[3:6]  and their cross product.
    For an AXIAL stack that is exactly (x, y, z) and the answer is [0, 1, 2].
    For any other axis-aligned stack it is a permutation: a SAGITTAL stack
    (IOP = 0,1,0, 0,0,-1) gives [1, 2, 0] - columns run along y, rows along z,
    and the slice stack along x.

    WHY IT MATTERS. The velocity components are in PATIENT axes (they are what
    Lambda acts on), but the flow-rate and flux tests slice the ARRAY along a
    chosen axis and read the velocity component with the SAME index. That is
    only correct when the two coincide. On a sagittal stack whose tube runs
    along x, the tube runs along array axis 2 while the through-plane velocity
    is component 0, so the test sliced the volume lengthwise through the tube
    and reported a flow rate that was wrong by ~15% with a plane-to-plane
    spread of ~40% (true value: 0.5%) - and nothing flagged it, because the
    guards only compare the two indices with each other. It never showed on
    the axial ideal phantom, where the two always coincide.

    Returns a list p with p[a] = patient axis of array axis a, or None when the
    stack is not axis-aligned within `tol_deg` (then no re-ordering is possible
    and the tests must not be trusted for that data).
    """
    if iop is None or len(iop) != 6:
        return None
    r = np.asarray(iop[0:3], float)
    c = np.asarray(iop[3:6], float)
    dirs = [r, c, np.cross(r, c)]
    p, sg, cos_tol = [], [], float(np.cos(np.radians(tol_deg)))
    for d in dirs:
        d = d / max(np.linalg.norm(d), 1e-12)
        a = int(np.argmax(np.abs(d)))
        if abs(d[a]) < cos_tol:
            return None
        p.append(a)
        sg.append(1 if d[a] > 0 else -1)
    if sorted(p) != [0, 1, 2]:
        return None
    return (p, sg) if return_signs else p


def reorient_to_patient(a, n_lead: int, perm, signs):
    """
    Put the three SPATIAL axes of `a` into patient order (x, y, z), flipping any
    axis whose index runs AGAINST its patient axis, so that index increases with
    the patient coordinate (which is what a VTK image with a positive spacing
    and an origin at the minimum coordinate requires).

    n_lead : number of non-spatial axes in front of the spatial ones
             (0 for (Nx,Ny,Nz), 1 for (3,Nx,Ny,Nz)). Trailing axes (time) stay last.
    perm   : perm[a] = patient axis of array axis a  (from array_axes_to_patient)
    signs  : signs[a] = +1 if array axis a increases along its patient axis
    """
    for ax in range(3):
        if signs[ax] < 0:
            a = np.flip(a, axis=n_lead + ax)
    tr = [int(i) for i in np.argsort(perm)]
    order = (list(range(n_lead)) + [n_lead + t for t in tr]
             + list(range(n_lead + 3, a.ndim)))
    return np.ascontiguousarray(np.transpose(a, order))


def dicom_flow_rate(v_field, lumen, spacing_m, axis: int):
    """
    Volumetric flow rate Q [ml/min] through planes normal to `axis`.

    Q = sum over the lumen cross-section of  v_normal * voxel_area.

    This is the only quantity in the whole pipeline that can be checked
    against something outside the MRI: a phantom's pump or flowmeter setting.
    Watanabe et al. (2019) validate 4D-flow this way, and for an incompressible
    steady flow Q must ALSO be the same through every plane along the tube —
    so the spread across planes is itself an internal consistency check, with
    no external reference needed.

    v_field : (3, X, Y, Z) in m/s (time-averaged if the data is 4D)
    lumen   : (X, Y, Z) bool, the flow channel
    spacing_m : (dx, dy, dz) in metres
    axis    : 0/1/2 — the through-plane direction
    Returns (per_plane_ml_min, plane_index) for planes with enough lumen.
    """
    dx, dy, dz = [float(t) for t in spacing_m]
    area = {0: dy * dz, 1: dx * dz, 2: dx * dy}[axis]      # m^2 per voxel
    vn = np.asarray(v_field, dtype=np.float64)[axis]        # m/s
    qs, idx = [], []
    n_planes = lumen.shape[axis]
    for k in range(n_planes):
        sl = [slice(None)] * 3
        sl[axis] = k
        m = lumen[tuple(sl)]
        if int(m.sum()) < 20:            # too small a cross-section to trust
            continue
        # m^3/s -> ml/min  (1 m^3 = 1e6 ml, 60 s per minute)
        qs.append(float(vn[tuple(sl)][m].sum() * area * 6.0e7))
        idx.append(k)
    return np.asarray(qs), np.asarray(idx)


def _pipe_axis_assignment(v, mask, iop, spacing_m, min_elongation=3.0,
                          min_flow_share=0.60, floor=0.30):
    """
    Decide from the data whether a pipe's velocity components sit on the right axes.

    The premise of the pipe preset is a STRAIGHT tube, and flow runs along a tube.
    So: find the largest connected object in the analysis mask, take the patient
    axis along which it is longest as the tube axis, and see which velocity
    component carries the speed inside it. If they differ, the assignment is wrong.

    Gates (dimensionless): the object must be clearly a tube (longest extent at
    least `min_elongation` times the next), one component must dominate
    (>= `min_flow_share` of the summed component speeds) and the tube axis must
    carry less than `floor` - the same floor flux_conservation enforces.

    v    : (3, X, Y, Z[, T]) velocity in PATIENT-axis components
    mask : (X, Y, Z) analysis mask in ARRAY axes
    iop  : ImageOrientationPatient (maps array axes to patient axes)
    Returns a dict (with "mismatch": bool) or None when it cannot say.
    """
    from scipy import ndimage as _ndi
    pm = array_axes_to_patient(iop)
    if pm is None:
        return None
    lab, n = _ndi.label(mask)
    if n == 0:
        return None
    sizes = _ndi.sum(mask, lab, index=range(1, n + 1))
    comp = lab == (int(np.argmax(sizes)) + 1)
    if int(comp.sum()) < 200:
        return None
    # Extent per axis, ROBUST to a single corrupted slice. The plain bounding box
    # (max - min of occupied indices) is exact for a clean tube, but one bad slice
    # bridging across the FOV via 3D connectivity - a zipper artefact, a stray
    # bright voxel run - dominates it: on a 0.67 mm pipe series, a single artefact
    # slice out of 160 (0.85% of the object's voxels) inflated the cross-axis
    # extent from 17 mm to 63 mm and pulled elongation from ~6x under 3x, which
    # silently skipped the axis swap the geometry clearly called for. Using the
    # 0.5th/99.5th percentile of each axis's OWN voxel coordinates (not a per-axis
    # bounding box of the other two) discounts any single plane that holds well
    # under 1% of the object's voxels, while leaving a true elongated tube's
    # extent - where every plane contributes - essentially unchanged.
    coords = np.nonzero(comp)                              # coords[a] = array-axis-a indices
    ext = [0.0, 0.0, 0.0]                                  # by PATIENT axis
    for a in range(3):
        c = coords[a]
        lo, hi = np.percentile(c, [0.5, 99.5])
        ext[pm[a]] = max(hi - lo, 1.0) * float(spacing_m[a]) * 1000.0
    order = np.argsort(ext)[::-1]
    g = int(order[0])
    elong = ext[g] / max(ext[int(order[1])], 1e-9)
    sp = [float(np.abs(v[c][comp]).mean()) for c in range(3)]
    tot = sum(sp) or 1.0
    shares = [x / tot for x in sp]
    f = int(np.argmax(shares))
    return {"tube_axis": g, "flow_component": f, "elongation": float(elong),
            "extent_mm": ext, "shares": shares,
            "is_tube": bool(elong >= min_elongation),
            "mismatch": bool(elong >= min_elongation and f != g
                             and shares[f] >= min_flow_share and shares[g] < floor)}


def _mask_background_diagnosis(mag_vol, inside, speed_tavg, venc_ms, frac_used):
    """
    Are the voxels that exceed VENC inside the signal mask BACKGROUND or OBJECT?

    In phase-contrast data the air/background has uniformly random phase, so
    about 48% of background voxels read |v| > VENC purely by geometry. If the
    signal mask lets background in, the two data-integrity alarms fire - "43%
    of voxels exceed VENC -> storage-order mismatch" and "mean |v| is 90% of
    VENC -> this is magnitude data" - although the phase data are perfectly
    good. They cannot tell the two causes apart; the magnitude can: split the
    masked voxels at the level that separates object from background (Otsu, on
    the non-zero voxels so a zero-filled unwarp border cannot capture the
    split) and compare the exceedance in the two groups.

    Returns a dict, or None when there is too little of either group to say.
    No threshold is assumed for the mask itself; the caller's alarm limits
    (2% / 10%) are reused for the verdict.
    """
    mv = np.asarray(mag_vol, dtype=np.float64)
    nz = mv[mv > 0]
    t = _otsu_threshold(nz) if nz.size else _otsu_threshold(mv)
    obj = inside & (mv > t)
    bg = inside & ~(mv > t)
    n_in, n_obj, n_bg = int(inside.sum()), int(obj.sum()), int(bg.sum())
    if n_in == 0 or n_obj < 50 or n_bg < 50:
        return None
    over = np.asarray(speed_tavg) > float(venc_ms)
    mmax = max(float(mv.max()), 1e-12)
    return {
        "otsu": float(t), "suggested_frac": float(t / mmax),
        "used_threshold": float(frac_used * mmax), "frac_used": float(frac_used),
        "kept_pct_of_volume": 100.0 * n_in / inside.size,
        "obj_pct_of_mask": 100.0 * n_obj / n_in,
        "over_all": float(over[inside].mean()),
        "over_obj": float(over[obj].mean()),
        "over_bg": float(over[bg].mean()),
        "background_in_mask": bool(over[obj].mean() <= 0.02 and over[bg].mean() > 0.10),
    }


def _print_mask_background_banner(d, preset_name):
    """Explain a background-in-mask verdict from _mask_background_diagnosis()."""
    print()
    print("  " + "#" * 68)
    print("  [DICOM] SIGNAL-MASK PROBLEM — not a phase-data or storage-order problem.")
    print(f"          {100*d['over_all']:.0f}% of the masked voxels exceed VENC, but they are BACKGROUND "
          "(random phase).")
    print(f"          The mask (magnitude > {100*d['frac_used']:.0f}% of the maximum = "
          f"{d['used_threshold']:.0f}, preset '{preset_name}') keeps "
          f"{d['kept_pct_of_volume']:.0f}% of the volume.")
    print(f"          Split at the level that separates object from background "
          f"(Otsu, {d['otsu']:.0f}):")
    print(f"            object voxels     {d['obj_pct_of_mask']:5.1f}% of the mask, "
          f"{100*d['over_obj']:4.1f}% exceed VENC   -> the phase data are fine")
    print(f"            background voxels {100-d['obj_pct_of_mask']:5.1f}% of the mask, "
          f"{100*d['over_bg']:4.1f}% exceed VENC   -> random phase")
    print(f"          Raise the mask fraction (start near {d['suggested_frac']:.2f}, the Otsu level as "
          "a fraction of the maximum):")
    print("            --mask-frac F   or the preset's DICOM_MAGNITUDE_MASK_FRAC.")
    print("          Choose F from a sweep: above the background, BELOW the dimmest part of the object.")
    print(f"          Also check the dataset preset in force ('{preset_name}') is the right type.")
    print("  " + "#" * 68)
    print()


def _print_axis_mismatch_banner(geo_axis, flow_axis, geo_share, flow_share,
                                direction_axes, flags_applied, permute):
    """
    A straight pipe runs along one axis; if the flow shows up in ANOTHER velocity
    component the three stored velocity frames are assigned to the wrong axes, and
    the flow rate, the flux test and the correction are all invalid (the correction
    applies the wrong rows of Lambda). Say so, and say what to set.

    direction_axes : the frame->axis string the loader applied ('xyz' = as stored)
    permute        : DICOM_DIRECTION_PERMUTE if it was applied, else None
    """
    g, v = "xyz"[geo_axis], "xyz"[flow_axis]
    frame = None
    try:
        # final component i <- loaded component po[i] <- acquired frame
        das = direction_axes or "xyz"
        po = [ "xyz".index(c) for c in permute ] if permute else [0, 1, 2]
        frame = das.index("xyz"[po[flow_axis]])
    except Exception:
        frame = None
    print()
    print("  " + "#" * 68)
    print("  [DICOM] AXIS MISMATCH — the velocity components are probably assigned to the "
          "wrong axes.")
    print(f"          The channel is a straight tube along {g} (only {100*geo_share:.0f}% of the "
          f"speed runs along it),")
    print(f"          but the flow is in velocity component {v} ({100*flow_share:.0f}%).")
    print("          Flow rate, flux conservation and the correction in THIS run are NOT valid:")
    print("          the correction would apply the wrong rows of Lambda.")
    if frame is not None:
        rest = iter([a for a in "xyz" if a != g])
        order = "".join(g if d == frame else next(rest) for d in range(3))
        print(f"          The flow was acquired as frame {frame} (frames 0, 1, 2 = the three velocity "
              "volumes in acquisition order).")
        print(f"          Re-run with:   --direction-order {order}"
              + ("   --no-axis-flags" if flags_applied else ""))
        print("          (the flow frame must encode the tube axis; the order of the other two frames "
              "cannot be")
        print("           determined from the data and changes the correction only slightly.)")
    else:
        print(f"          Set --direction-order so that the frame carrying the flow encodes '{g}'.")
    if flags_applied:
        print("          NOTE: axis flags (DICOM_DIRECTION_PERMUTE / _SIGNS) are being applied "
              "to this run.")
        print("          If they were derived on another acquisition, drop --direction-permute / "
              "--direction-signs (or pass --no-axis-flags).")
    print("  " + "#" * 68)
    print()


def _otsu_threshold(vol) -> float:
    """
    Otsu's threshold: the level that best splits the histogram into two
    populations (air and object) by maximising between-class variance.

    Used as a fallback when a fixed fraction of the maximum fails, which
    happens when the magnitude image does not use its dynamic range — the
    fraction is then anchored to a peak that carries no information about
    where the air/object boundary actually lies. Otsu asks the data instead.
    """
    v = np.asarray(vol, dtype=np.float64).ravel()
    v = v[np.isfinite(v)]
    if v.size == 0:
        return 0.0
    lo, hi = float(v.min()), float(v.max())
    if hi <= lo:
        return lo
    hist, edges = np.histogram(v, bins=256, range=(lo, hi))
    centres = 0.5 * (edges[:-1] + edges[1:])
    w = np.cumsum(hist).astype(np.float64)          # class-0 weight
    w1 = w[-1] - w                                   # class-1 weight
    csum = np.cumsum(hist * centres)
    with np.errstate(invalid="ignore", divide="ignore"):
        m0 = csum / w
        m1 = (csum[-1] - csum) / w1
        between = w * w1 * (m0 - m1) ** 2
    between[~np.isfinite(between)] = -np.inf
    return float(centres[int(np.argmax(between))])


def _fig_label(orig: int) -> str:
    """
    Console tag for a figure, using the SAME number as the saved filename.

    Log lines used to be hardcoded with the ORIGINAL figure numbers ([fig12],
    [fig08], "figs 07/08/12/13") while the files on disk are renumbered to
    fig01..fig05 by FIG_RENUMBER. So the console referred to figures the user
    could not find, and to figure numbers that mean something different on
    disk. Routing every label through here keeps the two in step, including
    if FIG_RENUMBER is edited.

    Figures that are computed but never written (their number is absent from
    FIG_RENUMBER) get a neutral tag instead of a phantom figure reference.
    """
    n = FIG_RENUMBER.get(int(orig))
    return f"[fig{n:02d}]" if n else "[metrics]"


def _json_safe(arr):
    """NaN -> None so json.dump emits valid `null` rather than bare NaN."""
    import math
    out = [None if (v is None or math.isnan(float(v))) else float(v)
           for v in (arr if len(arr) else [])]
    if not out:
        return None
    return out[0] if len(out) == 1 else out


def _fig_list(*origs: int) -> str:
    """'fig02/fig04' for the figures among origs that are actually saved."""
    kept = sorted(FIG_RENUMBER[o] for o in origs if o in FIG_RENUMBER)
    return "/".join(f"fig{n:02d}" for n in kept) if kept else "the saved figures"

# Vessel-signal threshold for the % error map (saved as fig02) and vessel metrics,
# expressed as a FRACTION of the peak true velocity.  The percentage error
# 100*|dv|/|v_true| blows up near the vessel wall and ends where |v_true| -> 0,
# producing large-percentage speckle that is a divide-by-small artefact rather
# than residual distortion.  Excluding voxels below this fraction of the peak
# removes that artefact and works across flow regimes (fast vessel or slow
# pipe).  Raise it for a stricter/cleaner map (fewer voxels); lower it to keep
# more of the vessel.  0.20 keeps roughly the inner ~80% of a parabolic lumen.
PCT_ERROR_VESSEL_FRACTION = 0.20

# Self-test outcomes, filled in by run_self_tests() and written into
# run_config.json. The console scrollback is not a reproducibility record: T5
# is the test the thesis quotes as non-circular evidence, so its numbers have
# to survive in the run log next to the results they support.
SELF_TEST_RESULTS: Dict[str, object] = {}

# ── SNR sensitivity sweep ────────────────────────────────────────────────────
# The synthetic percentage improvement is dominated by the assumed SNR (the
# residual after correction is essentially pure noise), so a single number is
# not quotable on its own. The sweep costs one extra correction pass per SNR —
# a few seconds on the default grid — and produces the table that belongs
# beside the headline in the validation chapter.
RUN_SNR_SWEEP: bool = True
SNR_SWEEP_DB: Tuple[float, ...] = (20.0, 25.0, 30.0, 40.0, 50.0)
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# Optional DICOM support
# ─────────────────────────────────────────────────────────────────────────────
try:
    import pydicom
    PYDICOM_AVAILABLE = True
except ImportError:
    PYDICOM_AVAILABLE = False
    print("[WARN] pydicom not found — DICOM pipeline disabled. "
          "Install via:  pip install pydicom")

# ─────────────────────────────────────────────────────────────────────────────
# GUI (tkinter — optional)
# ─────────────────────────────────────────────────────────────────────────────
#
# OPTIONAL ON PURPOSE. tkinter ships with Python on Windows but is a separate
# package on most Linux distributions and is absent from typical CI images and
# headless servers. Importing it at module level made the whole toolbox
# unimportable there — including `--help`, which failed before argparse ever
# ran. Since the goal is a toolbox others can use on their own platform, the
# GUI is now a convenience that degrades to the command line rather than a
# hard dependency.
try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
    TK_AVAILABLE = True
except ImportError:                       # headless / no tkinter installed
    tk = None
    filedialog = messagebox = ttk = None
    TK_AVAILABLE = False

# Suppress only the cosmetic warnings this pipeline provokes deliberately
# (empty-slice reductions on fully masked volumes, /0 inside np.where
# guards). Genuine numerical warnings are NOT hidden: a blanket
# filterwarnings("ignore") would also mask LinAlgWarning from an
# ill-conditioned Lambda inversion, which must stay visible.
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ─────────────────────────────────────────────────────────────────────────────
# Global Matplotlib style
# ─────────────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "font.size":        9,
    "axes.titlesize":   9,
    "axes.labelsize":   8,
    "xtick.labelsize":  7,
    "ytick.labelsize":  7,
    "legend.fontsize":  7,
    "figure.dpi":       130,
    "axes.linewidth":   0.6,
    "grid.linewidth":   0.4,
    "grid.alpha":       0.5,
})

PALETTE = {
    "true":       "#2C2C2A",
    "std":        "#E24B4A",
    "corrected":  "#1D9E75",
    "accent_violet": "#6A5ACD",
    "accent_amber":  "#E8A838",
    "highlight":  "#3A7EBF",
}


# ═════════════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class GradientCoefficients:
    """Parsed manufacturer gradient coefficient data."""
    filepath:   str
    R0:         float                         # Reference radius [m]
    coeffs:     Dict[str, List[Tuple]]        # axis → [(type, n, m, val), …]
    coil_name:  str = "Unknown"
    max_degree: int = 0
    lnorm:      int = 4


@dataclass
class SimulationConfig:
    """All tunable simulation parameters in one place."""
    # Spatial grid
    # ODD on purpose. An even grid_N has no sample at rho = 0, so the peak of
    # the Gaussian vessel profile is never sampled and every % error is
    # referenced to an under-read truth: at grid_N = 48 the on-grid peak was
    # 66.6 cm/s against the configured 80 cm/s, a 16.7% shortfall, and fig01
    # had to be drawn at y = +4.68 mm because no plane lay at isocentre.
    # grid_N = 97 gives 5.5 voxels across the FWHM, samples y = 0.000 mm
    # exactly, and reduces the shortfall to 0.0%. Lambda then takes ~86 s
    # instead of ~13 s. Use 143 (8.1 vox/FWHM, ~5 min) if runtime allows.
    grid_N:     int   = 97          # Voxels per axis — keep ODD (see above)
    r_max_m:    float = 0.22        # Half-FOV [m]  (±220 mm)
    # 4-D temporal
    Nt:         int   = 17          # Cardiac phases (matches acquired data)
    # Tube phantom (Mode 1)
    # The physical pipe phantom has a lumen radius of ~8 mm, but on the
    # default 97³ grid spanning ±220 mm the voxel edge is ~4.6 mm, so that
    # tube would be under 4 voxels across. 30 mm is used here so the
    # diameter spans ~13 voxels (voxel/diameter ≈ 8 %, inside the 10 %
    # criterion of Fukuyama 2017) and flow integration is meaningful.
    tube_radius_m:  float = 0.030   # 30 mm simulation tube radius
    tube_length_m:  float = 0.150   # 150 mm length
    # Mode 1 temporal behaviour. False (default): STEADY flow, every cardiac
    # phase identical — the proxy for the steady-flow pipe phantom, so the
    # time-averaged and single-frame vessel sets coincide. True: the earlier
    # behaviour, profile scaled by 0.85 + 0.15·waveform (85–100 % of v_max);
    # kept only so synthetic results produced before v1.0 can be reproduced
    # (command line: --mode1-pulsatile).
    mode1_pulsatile: bool = False
    # NOTE: this is an OUTPUT, not a tunable input. generate_poiseuille_4d
    # overwrites it with the flow rate implied by poiseuille_v_peak_m_s and
    # tube_radius_m. It is initialised to 0.0 so that a stale default can
    # never be reported as if it were the simulated flow rate.
    flow_rate_ml_min: float = 0.0   # [ml/min] computed, see generate_poiseuille_4d
    # Vessel simulation (Mode 2)
    # Gaussian FWHM raised to 25 mm so the profile is resolved by ≥ 5 voxels.
    vessel_fwhm_m:   float = 0.025  # Gaussian vessel FWHM [m]
    v_peak_systole:  float = 0.80   # Peak systolic velocity [m/s]
    # Mode 1 Poiseuille peak velocity [m/s]. Set to an acquisition-realistic
    # value (comparable to VENC and to real pipe-flow phantoms, ~tens of cm/s)
    # so the gradient-nonlinearity distortion sits well above the PC-MRI
    # velocity-noise floor (which is set by VENC, not by the flow speed).
    # Deriving v_max from Q=300 ml/min through the coarse-grid 30 mm tube gives
    # only ~0.35 cm/s, far below the noise, which makes the correction
    # untestable in simulation. See generate_poiseuille_4d.
    poiseuille_v_peak_m_s: float = 0.80
    VENC:            float = 0.90   # [m/s]
    # Flow pattern (Mode 2 only)
    # "axial" — purely axial Gaussian flow (default)
    # "helical" — axial + physiological tangential swirl in y-z plane
    # This mode also serves as the simulation proxy
    # for the Ideal Aneurysm Phantom dataset :
    # aneurysmal sac flow is dominated by swirl + shear, which
    # the helical pattern captures without prescribing a
    # specific saccular geometry.
    flow_pattern:    str   = "axial"
    helical_swirl:   float = 0.15   # Swirl fraction of peak velocity (≈15%)
    # Noise
    # The simulation noise level. It dominates the percentage improvement in
    # fig02/fig04, because after correction the residual is essentially pure
    # noise, so the same correction reads very differently at 50 dB and at
    # 20 dB. The SNR sweep (SNR_SWEEP_DB) prints the actual dependence.
    #
    # Do not guess it. The DICOM stage measures the velocity noise floor of
    # whatever data you load and prints the matching value, from
    #     SNR_dB = 20*log10( sqrt(2)*VENC / (pi*sigma_v) )
    # warning you if this setting disagrees by more than 3 dB. Pass it with
    # --snr-db, or quote the SNR-independent noise-free row instead.
    #
    # 50 dB (sigma_v = 0.13 cm/s at VENC 0.9) is an idealisation no 4D flow
    # acquisition achieves; it is the default only so the pipeline runs
    # without data.
    snr_db:          float = 50.0   # Signal-to-noise ratio [dB] — SEE ABOVE
    # SNR rationale and its effect on the fig02 residual.
    # The corrected % error is the SUM of two parts: the residual GNL
    # distortion (which the correction removes, -> ~0) and the phase noise
    # (which NO gradient-nonlinearity correction can remove). The noise floor
    # is sigma_v = sqrt(2)*VENC/(pi * SNR_lin); at the default VENC = 0.9 m/s:
    #     30 dB -> 1.28 cm/s (1.6 % of an 80 cm/s peak)
    #     40 dB -> 0.40 cm/s (0.5 %)
    #     50 dB -> 0.13 cm/s (0.16 %)  <- default
    #     60 dB -> 0.04 cm/s (0.05 %)
    # Once sigma_v approaches the systematic GNL error on the vessel voxels,
    # the corrected residual is dominated by irreducible noise and the
    # improvement metric falls, although the correction itself is unchanged;
    # without noise the improvement is 100 % to machine precision. Report the
    # SNR alongside any improvement figure, and quote the noise-free row and
    # the SNR sweep (SNR_SWEEP_DB, printed to the console and stored in
    # run_config.json — no separate figure is written).
    add_noise:       bool  = True
    # Numerical differentiation step for Λ
    eps_m:           float = 5e-4   # [m]  ≈ 0.5 mm
    # Invertibility gate for Λ⁻¹: a voxel is corrected only when the
    # smallest singular value of Λ exceeds this. 1/σ_min is the worst-case
    # noise amplification of the inversion, so σ_min — not det(Λ) = σ1·σ2·σ3,
    # which can stay O(1) while σ_min → 0 — is the quantity that measures
    # invertibility. Voxels failing the gate keep the identity inverse and
    # are reported in VelocityCorrector.last_valid_mask.
    sigma_min_threshold: float = 1e-4
    # DEPRECATED — former determinant-based singularity threshold; kept so
    # existing configs still parse, no longer consulted by the corrector.
    det_threshold:   float = 1e-8
    # Tikhonov regularisation strength for Λ⁻¹ noise suppression.
    # Replaces the exact inverse Λ⁻¹ with the regularised pseudoinverse
    # (Λᵀ·Λ + λ²·I)⁻¹ · Λᵀ
    # which dampens noise amplification at high-κ voxels (FOV edges).
    # Set to 0.0 to disable. Typical effective values: 0.02 – 0.10.
    # Trades a small forward bias for a large variance reduction; benefits
    # the smallest signal components (vy, vz) the most.
    tikhonov_lambda: float = 0.05
    # Adaptive Tikhonov: when True, λ varies per voxel with κ(Λ) as a linear
    # ramp between the two thresholds below:
    #     λ_local = tikhonov_lambda · clip((κ − κ_low)/(κ_high − κ_low), 0, 1)
    # so voxels with κ ≤ κ_low get the exact inverse (no bias) and voxels
    # with κ ≥ κ_high get the full λ. Default True. This ramp is a custom
    # heuristic (see "Known deviations").
    tikhonov_adaptive: bool = True
    # κ-threshold for adaptive Tikhonov: voxels with κ ≤ κ_low get exact
    # inverse (λ=0), voxels with κ ≥ κ_high get full regularisation.
    tikhonov_kappa_low:  float = 2.0
    tikhonov_kappa_high: float = 10.0
    # ── field strength and synthetic error-injection parameters ─────────────
    # Main field strength [T] (auto-detected from coil identifier in main();
    # 3T for Skyra; 7T for both SC72CD (7T+) and IMPULSE/Athena (7Tx)).
    # Retained for the coil-vs-scanner consistency check (the header field
    # strength must match the coil the .grad file belongs to) and for
    # reporting; no field-error model here depends on it.
    B0:              float = 3.0
    cs_phase_omit:      bool  = False
    # Spherical-harmonic normalisation convention for the m>0 terms.
    #   "schmidt"    — Schmidt quasi-normalisation sqrt((2-d_m0)(n-m)!/(n+m)!)
    #                  (this code's original convention, and the one that is
    #                  SELF-CONSISTENT with the coefficient file: with
    #                  Schmidt and NO Condon-Shortley phase, a unit primary
    #                  coefficient A(1,1) evaluates to exactly +x/R0, i.e.
    #                  the nominal gradient — which is what the file header
    #                  "A(1,0) = B(1,1) = A(1,1) = 0" presupposes when it
    #                  zeroes the primaries).
    #   "gradunwarp" — the normalisation used by the HCP-maintained
    #                  gradunwarp reference for Siemens files: identical for
    #                  m = 0 (the whole z-coil matches to machine precision),
    #                  but sqrt(2n+1)/2 LARGER per m>0 term, so the x/y coil
    #                  perturbations grow by 32% (n=3) to 3.3x (n=21).
    # The two disagree by up to ~28% of the x/y perturbation inside R0
    # (see validate_lambda_vs_gradunwarp.py, shipped alongside). Which one
    # Siemens actually used cannot be proven from the file alone; the
    # Lambda-independent flux-conservation check on acquired data is the
    # built-in discriminator: run the DICOM pipeline once per setting and
    # keep the convention that conserves mass better.
    sh_normalization:   str   = "schmidt"


@dataclass
class ValidationResults:
    """Container for all computed validation metrics."""
    rmse_before:    np.ndarray = field(default_factory=lambda: np.array([]))
    rmse_after:     np.ndarray = field(default_factory=lambda: np.array([]))
    mae_before:     np.ndarray = field(default_factory=lambda: np.array([]))
    mae_after:      np.ndarray = field(default_factory=lambda: np.array([]))
    pearson_before: np.ndarray = field(default_factory=lambda: np.array([]))
    pearson_after:  np.ndarray = field(default_factory=lambda: np.array([]))
    flow_rate_ref:  float      = 0.0
    flow_rate_std:  float      = 0.0
    flow_rate_cor:  float      = 0.0
    pct_improvement: float     = 0.0
    # Bootstrap 95 % confidence intervals on RMSE
    rmse_before_ci: Tuple[float, float] = (0.0, 0.0)
    rmse_after_ci:  Tuple[float, float] = (0.0, 0.0)
    pct_improvement_ci: Tuple[float, float] = (0.0, 0.0)
    # Whole-sphere (r ≤ R0) diagnostics, kept separate from the
    # vessel-region headline so the two are never conflated. Over the whole
    # sphere the metric is dominated by Λ⁻¹ noise amplification in empty edge
    # voxels and is NOT the clinical figure of merit.
    pct_improvement_whole:  float = 0.0
    rmse_before_whole:      float = 0.0
    rmse_after_whole:       float = 0.0
    rmse_before_ci_whole: Tuple[float, float] = (0.0, 0.0)
    rmse_after_ci_whole:  Tuple[float, float] = (0.0, 0.0)
    # Vessel-masked metrics for the fig12 summary. The threshold is
    # relative (PCT_ERROR_VESSEL_FRACTION x peak |v_true|), not a fixed
    # 5 cm/s, so it adapts to the flow regime of each mode.
    # consistent with fig07/fig08/fig09.
    vessel_n:               int        = 0
    rmse_before_vessel:     float      = 0.0
    rmse_after_vessel:      float      = 0.0
    mae_before_vessel:      float      = 0.0
    mae_after_vessel:       float      = 0.0
    p95_before_vessel:      float      = 0.0
    p95_after_vessel:       float      = 0.0
    pearson_before_vessel:  np.ndarray = field(default_factory=lambda: np.array([]))
    pearson_after_vessel:   np.ndarray = field(default_factory=lambda: np.array([]))

    # ── Metrics that separate the SYSTEMATIC error from the NOISE ────────
    #
    # WHY THESE EXIST. Every metric above is computed on the TIME-AVERAGED
    # field. Gradient nonlinearity is systematic and survives that average;
    # thermal noise does not (it falls by sqrt(Nt)). After correction the
    # residual is essentially pure noise, so the averaging shrinks the
    # "after" term and not the "before" term and the percentage improvement
    # inflates. The same headline therefore moves from ~92% at SNR = 50 dB
    # to ~5% at SNR = 20 dB WITHOUT the correction changing at all.
    #
    # The three fields below make that separable in the thesis:
    #   *_perframe            - the same RMSE without temporal averaging
    #   rmse_systematic_*     - the run repeated with the noise switched off,
    #                           i.e. the pure GNL error and the pure residual.
    #                           This is the number that actually characterises
    #                           the correction; it is independent of SNR.
    #   snr_db                - recorded so no figure or table can be quoted
    #                           without the noise level it assumed.
    rmse_before_vessel_perframe: float = 0.0
    rmse_after_vessel_perframe:  float = 0.0
    pct_improvement_perframe:    float = 0.0
    rmse_systematic_before_vessel: float = 0.0
    rmse_systematic_after_vessel:  float = 0.0
    pct_improvement_systematic:    float = 0.0
    snr_db:                        float = 0.0
    # Discretisation diagnostics for the synthetic vessel (see M3): the peak
    # of a Gaussian profile on a coarse even grid is under-read, so every
    # percentage error is referenced to a partial-volumed truth.
    v_peak_configured_m_s: float = 0.0
    v_peak_on_grid_m_s:    float = 0.0
    voxels_per_fwhm:       float = 0.0


# ═════════════════════════════════════════════════════════════════════════════
# GUI FILE / DIRECTORY SELECTION
# ═════════════════════════════════════════════════════════════════════════════

class PathSelectorGUI:
    """
    Windows-compatible tkinter GUI for selecting:
      · gradient coefficient file (.grad / .txt)
      · DICOM directory (optional)
      · output directory
    """

    def __init__(self):
        self.grad_file:   Optional[str] = None
        self.dicom_dir:   Optional[str] = None
        self.output_dir:  Optional[str] = None
        self.mag_dir:     Optional[str] = None   # optional magnitude/mask folder
        self.venc_override: Optional[float] = None  # optional VENC override [m/s]
        # Optional per-run axis mapping (same meaning and validation as
        # --direction-permute / --direction-signs). None = axes as loaded.
        self.direction_permute: Optional[str] = None
        self.direction_signs: Optional[Tuple[int, int, int]] = None
        self.dataset_preset: str = DATASET_PRESET   # in_vivo_head / phantom / manual
        self.sim_mode:    int           = 1     # 1=Poiseuille, 2=Pulsatile, 3=Both
        self.flow_pattern: str          = "axial"   # "axial" or "helical"
        self.run_dicom:   bool          = False

    # ── internal helpers ──────────────────────────────────────────────────

    @staticmethod
    def _make_root() -> "tk.Tk":     # quoted: evaluated lazily, so this class
        #                              body imports fine without tkinter
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        return root

    @staticmethod
    def _pick_file(title: str, ftypes: list) -> Optional[str]:
        root = PathSelectorGUI._make_root()
        path = filedialog.askopenfilename(
            title=title,
            filetypes=ftypes,
            initialdir=str(Path.home()),
        )
        root.destroy()
        return path if path else None

    @staticmethod
    def _pick_dir(title: str) -> Optional[str]:
        """
        Pick a directory using askdirectory.
        Used for the OUTPUT folder (where figures are saved).
        """
        root = PathSelectorGUI._make_root()
        path = filedialog.askdirectory(
            title=title,
            initialdir=str(Path.home()),
        )
        root.destroy()
        return path if path else None

    @staticmethod
    def _pick_dicom(title: str) -> Optional[str]:
        """
        Pick DICOM files using askopenfilenames (multi-select enabled).

        The user can:
          - Select ALL 4 files at once with Ctrl+A
          - Select multiple files with Ctrl+click
          - Select just ONE file — the program automatically uses
            the containing folder to find all sibling .dcm files

        Returns the folder path containing the selected file(s).
        """
        root = PathSelectorGUI._make_root()
        paths = filedialog.askopenfilenames(
            title=title,
            initialdir=str(Path.home()),
            filetypes=[
                # 'All files' FIRST so it is the DEFAULT filter — many 4D-flow
                # exports (e.g. Siemens 'MR.1.3.12…') have NO .dcm extension and
                # would be hidden by a *.dcm filter ("No items match your
                # search"). find_dicoms() validates real DICOMs by magic bytes.
                ("All files",    "*.*"),
                ("DICOM files",  "*.dcm *.DCM *.IMA *.ima"),
            ],
            multiple=True,
        )
        root.destroy()
        if not paths:
            return None
        # Return the parent folder — DicomLoader will scan it for all DICOM
        # files automatically (selecting even ONE file is enough).
        return str(Path(paths[0]).parent)

    # ── main launcher ─────────────────────────────────────────────────────

    def launch(self) -> bool:
        """
        Show the master selection dialog.
        Returns True if the user confirmed, False if cancelled.
        """
        root = tk.Tk()
        root.title("4D Flow MRI — Simulation Setup")
        root.resizable(True, True)

        # ── Header ────────────────────────────────────────────────────────
        hdr = tk.Frame(root, bg="#1A3A5C", pady=10)
        hdr.pack(fill="x")
        tk.Label(hdr,
                 text="4D Flow MRI  |  Gradient Nonlinearity Simulation",
                 bg="#1A3A5C", fg="white",
                 font=("Segoe UI", 13, "bold")).pack()
        tk.Label(hdr,
                 text="Open-Source Preprocessing Toolbox  ·  Master Thesis",
                 bg="#1A3A5C", fg="#88AACC",
                 font=("Segoe UI", 9)).pack()

        # ── Footer (packed BEFORE the scroll area so the Run button is always
        #    visible, no matter how tall the form grows or how small the screen)
        footer = tk.Frame(root, padx=18, pady=10)
        footer.pack(side="bottom", fill="x")

        # ── Scrollable body ───────────────────────────────────────────────
        # The form is taller than some laptop screens, so it lives on a canvas
        # with a vertical scrollbar and mouse-wheel support.
        _outer = tk.Frame(root)
        _outer.pack(side="top", fill="both", expand=True)
        _canvas = tk.Canvas(_outer, highlightthickness=0)
        _vbar = tk.Scrollbar(_outer, orient="vertical", command=_canvas.yview)
        _canvas.configure(yscrollcommand=_vbar.set)
        _vbar.pack(side="right", fill="y")
        _canvas.pack(side="left", fill="both", expand=True)

        body = tk.Frame(_canvas, padx=18, pady=14)
        _win = _canvas.create_window((0, 0), window=body, anchor="nw")

        def _on_body_config(_evt=None):
            _canvas.configure(scrollregion=_canvas.bbox("all"))
        body.bind("<Configure>", _on_body_config)

        def _on_canvas_config(evt):
            _canvas.itemconfigure(_win, width=evt.width)   # stretch to width
        _canvas.bind("<Configure>", _on_canvas_config)

        def _on_wheel(evt):
            delta = evt.delta
            _canvas.yview_scroll(int(-delta / 120) if delta else 0, "units")
        root.bind_all("<MouseWheel>", _on_wheel)                     # Windows
        root.bind_all("<Button-4>",
                      lambda e: _canvas.yview_scroll(-1, "units"))   # Linux
        root.bind_all("<Button-5>",
                      lambda e: _canvas.yview_scroll(1, "units"))

        row = [0]
        def next_row():
            r = row[0]; row[0] += 1; return r

        # ── Gradient file ─────────────────────────────────────────────────
        tk.Label(body, text="Gradient coefficient file (.grad / .txt):",
                 font=("Segoe UI", 9, "bold")).grid(
            row=next_row(), column=0, columnspan=3,
            sticky="w", pady=(6, 0))

        grad_var = tk.StringVar(value="<no file selected>")

        def _pick_grad():
            p = self._pick_file(
                "Select Gradient Coefficient File",
                [("Grad / Text files", "*.grad *.txt"),
                 ("All files", "*.*")])
            if p:
                self.grad_file = p
                grad_var.set(os.path.basename(p))
                grad_lbl.config(fg="#1A6A1A")

        r = next_row()
        tk.Button(body, text="Browse …", width=10, command=_pick_grad).grid(
            row=r, column=0, sticky="w", pady=2)
        grad_lbl = tk.Label(body, textvariable=grad_var,
                             fg="#8A0000", font=("Consolas", 8))
        grad_lbl.grid(row=r, column=1, columnspan=2, sticky="w", padx=8)

        # ── DICOM directory ───────────────────────────────────────────────
        tk.Label(body,
                 text="DICOM series — Browse → open the series folder → select any file (Ctrl+A = all). .dcm extension not required:",
                 font=("Segoe UI", 9, "bold")).grid(
            row=next_row(), column=0, columnspan=3,
            sticky="w", pady=(10, 0))

        dicom_var   = tk.StringVar(value="<not selected  (simulation-only mode)>")
        dicom_check = tk.BooleanVar(value=False)

        def _pick_dicom():
            p = self._pick_dicom(
                "Open the DICOM series folder and select any file "
                "(Ctrl+A selects all) — the whole folder is loaded")
            if p:
                self.dicom_dir = p
                # Show full folder path so user can verify it is correct.
                # Count ALL files (extensionless DICOMs like 'MR.1.3…' included)
                # so the display never wrongly reads 0.
                n_dcm = sum(1 for f in Path(p).iterdir() if f.is_file())
                dicom_var.set(f"{p}  [{n_dcm} file(s) in folder]")
                dicom_lbl.config(fg="#1A6A1A")
                dicom_check.set(True)

        r = next_row()
        tk.Button(body, text="Browse …", width=10, command=_pick_dicom).grid(
            row=r, column=0, sticky="w", pady=2)
        dicom_lbl = tk.Label(body, textvariable=dicom_var,
                              fg="#555555", font=("Consolas", 8))
        dicom_lbl.grid(row=r, column=1, columnspan=2, sticky="w", padx=8)

        # ── Magnitude / mask (optional): a single FILE or a folder ────────
        tk.Label(body,
                 text="Magnitude for signal mask (optional) — pick the magnitude FILE (e.g. ww25_Magnitude…dcm) OR a _MAG folder:",
                 font=("Segoe UI", 9, "bold")).grid(
            row=next_row(), column=0, columnspan=3,
            sticky="w", pady=(10, 0))

        mag_var = tk.StringVar(value="<none  (mask skipped — correction still runs)>")

        def _pick_mag_file():
            p = self._pick_file("Select Magnitude DICOM File (optional)",
                                 [("DICOM files", "*.dcm *.ima *.IMA"),
                                  ("All files", "*.*")])
            if p:
                self.mag_dir = p
                mag_var.set(p)
                mag_lbl.config(fg="#1A6A1A")

        def _pick_mag_folder():
            p = self._pick_dir("Select Magnitude / Mask Folder (optional)")
            if p:
                self.mag_dir = p
                mag_var.set(p)
                mag_lbl.config(fg="#1A6A1A")

        r = next_row()
        tk.Button(body, text="Browse file \u2026", width=13,
                  command=_pick_mag_file).grid(row=r, column=0, sticky="w", pady=2)
        tk.Button(body, text="Browse folder \u2026", width=14,
                  command=_pick_mag_folder).grid(row=r, column=1, sticky="w", pady=2)
        mag_lbl = tk.Label(body, textvariable=mag_var,
                           fg="#555555", font=("Consolas", 8))
        mag_lbl.grid(row=next_row(), column=0, columnspan=3, sticky="w", padx=8)

        # ── Dataset type ──────────────────────────────────────────────────
        # The head and the phantoms need different masking and display choices,
        # and getting them crossed produces silently wrong output (a brain mask
        # erodes a thin pipe lumen). Selecting here means neither dataset's
        # settings have to be edited into the file, so switching between them
        # cannot leave anything behind.
        tk.Label(body,
                 text="Dataset type — selects the masking and display settings "
                      "for this run:",
                 font=("Segoe UI", 9, "bold")).grid(
            row=next_row(), column=0, columnspan=3, sticky="w", pady=(10, 0))
        _preset_labels = {
            "in vivo — brain / Circle of Willis": "in_vivo_head",
            "ideal phantom — cardiac or steady": "ideal_phantom",
            "pipe flow rig — steady or pulsatile": "pipe_flow",
            "manual — use the settings in the file": "manual",
        }
        _inv_preset = {v: k for k, v in _preset_labels.items()}
        preset_var = tk.StringVar(
            value=_inv_preset.get(DATASET_PRESET,
                                  "in vivo — brain / Circle of Willis"))
        r = next_row()
        tk.Label(body, text="Dataset:", font=("Segoe UI", 9)).grid(
            row=r, column=0, sticky="w", pady=2)
        tk.OptionMenu(body, preset_var, *_preset_labels.keys()).grid(
            row=r, column=1, columnspan=2, sticky="w", padx=8)

        # ── VENC override (optional) ──────────────────────────────────────
        tk.Label(body,
                 text="VENC override (optional) — if the VENC is not present in the "
                      "filename/header, type it here in m/s; blank = auto-detect:",
                 font=("Segoe UI", 9, "bold")).grid(
            row=next_row(), column=0, columnspan=3, sticky="w", pady=(10, 0))
        venc_var = tk.StringVar(value="")
        r = next_row()
        tk.Label(body, text="VENC [m/s]:", font=("Segoe UI", 9)).grid(
            row=r, column=0, sticky="w", pady=2)
        tk.Entry(body, textvariable=venc_var, width=12).grid(
            row=r, column=1, sticky="w", padx=8)

        # ── Axis mapping (optional) ───────────────────────────────────────
        # Same as --direction-permute / --direction-signs. Without these the
        # dialog could only ever run a dataset with its axes as loaded.
        tk.Label(body,
                 text="Velocity axis mapping (optional) — blank = axes as loaded. "
                      "A wrong mapping does not show in |v| but corrupts the "
                      "correction:",
                 font=("Segoe UI", 9, "bold")).grid(
            row=next_row(), column=0, columnspan=3, sticky="w", pady=(10, 0))
        perm_var  = tk.StringVar(value="")
        signs_var = tk.StringVar(value="")
        r = next_row()
        tk.Label(body, text="Permute (e.g. yxz):", font=("Segoe UI", 9)).grid(
            row=r, column=0, sticky="w", pady=2)
        tk.Entry(body, textvariable=perm_var, width=12).grid(
            row=r, column=1, sticky="w", padx=8)
        r = next_row()
        tk.Label(body, text="Signs (e.g. -1,-1,1):", font=("Segoe UI", 9)).grid(
            row=r, column=0, sticky="w", pady=2)
        tk.Entry(body, textvariable=signs_var, width=12).grid(
            row=r, column=1, sticky="w", padx=8)
        tk.Label(body,
                 text="Thesis datasets:  Circle of Willis 7 T → yxz and -1,-1,1   ·   "
                      "aneurysm phantom 7 T → yxz and 1,1,-1   ·   pipe 3 T → blank",
                 font=("Segoe UI", 8), fg="#555555").grid(
            row=next_row(), column=0, columnspan=3, sticky="w")


        # ── Output directory ──────────────────────────────────────────────
        tk.Label(body, text="Output directory for figures:",
                 font=("Segoe UI", 9, "bold")).grid(
            row=next_row(), column=0, columnspan=3,
            sticky="w", pady=(10, 0))

        out_var = tk.StringVar(value="<no folder selected>")

        def _pick_out():
            p = self._pick_dir("Select Output Directory")
            if p:
                self.output_dir = p
                out_var.set(p)
                out_lbl.config(fg="#1A6A1A")

        r = next_row()
        tk.Button(body, text="Browse …", width=10, command=_pick_out).grid(
            row=r, column=0, sticky="w", pady=2)
        out_lbl = tk.Label(body, textvariable=out_var,
                            fg="#8A0000", font=("Consolas", 8))
        out_lbl.grid(row=r, column=1, columnspan=2, sticky="w", padx=8)

        # ── Simulation mode ───────────────────────────────────────────────
        tk.Label(body, text="Simulation mode:",
                 font=("Segoe UI", 9, "bold")).grid(
            row=next_row(), column=0, columnspan=3,
            sticky="w", pady=(14, 0))

        mode_var = tk.IntVar(value=1)   # default; select Mode 2 in the window for synthetic validation
        r = next_row()
        tk.Radiobutton(body, text="Mode 1 — Poiseuille tube phantom"
                       "  (steady, v_max = 80 cm/s, r = 30 mm simulation tube)",
                       variable=mode_var, value=1,
                       font=("Segoe UI", 9)).grid(
            row=r, column=0, columnspan=3, sticky="w")
        r = next_row()
        tk.Radiobutton(body, text="Mode 2 — Pulsatile cerebrovascular vessel"
                       "  (17 cardiac phases, v_peak = 0.8 m/s)",
                       variable=mode_var, value=2,
                       font=("Segoe UI", 9)).grid(
            row=r, column=0, columnspan=3, sticky="w")
        r = next_row()
        tk.Radiobutton(body, text="Mode 3 — Both modes (full validation suite)",
                       variable=mode_var, value=3,
                       font=("Segoe UI", 9)).grid(
            row=r, column=0, columnspan=3, sticky="w")

        # ── Flow pattern (Mode 2 / Mode 3 only) ──────────────────────────
        tk.Label(body,
                 text="Flow pattern for Mode 2  (helical = axial + swirl, "
                      "more physiological):",
                 font=("Segoe UI", 9, "bold")).grid(
            row=next_row(), column=0, columnspan=3,
            sticky="w", pady=(10, 0))

        pattern_var = tk.StringVar(value="axial")
        r = next_row()
        tk.Radiobutton(body, text="Axial only  (straight parabolic / Gaussian flow)",
                       variable=pattern_var, value="axial",
                       font=("Segoe UI", 9)).grid(
            row=r, column=0, columnspan=3, sticky="w")
        r = next_row()
        tk.Radiobutton(body,
                       text="Helical  (axial + 15 % tangential swirl  — "
                            "models aortic / cerebrovascular secondary flow)",
                       variable=pattern_var, value="helical",
                       font=("Segoe UI", 9)).grid(
            row=r, column=0, columnspan=3, sticky="w")

        # ── Separator & run button ────────────────────────────────────────
        ttk.Separator(body, orient="horizontal").grid(
            row=next_row(), column=0, columnspan=3,
            sticky="ew", pady=12)

        confirmed = [False]

        def _confirm():
            if not self.grad_file:
                messagebox.showerror("Missing input",
                                     "Please select a gradient coefficient file.")
                return
            if not self.output_dir:
                messagebox.showerror("Missing output",
                                     "Please select an output directory.")
                return
            self.dataset_preset = _preset_labels[preset_var.get()]
            _vtxt = venc_var.get().strip()
            if _vtxt:
                try:
                    self.venc_override = float(_vtxt)
                except ValueError:
                    messagebox.showerror(
                        "Invalid VENC",
                        "VENC override must be a number in m/s (e.g. 1.5), or blank.")
                    return
                # The field is in m/s. Typing the cm/s value (150 instead of
                # 1.5) is an easy slip and would scale every velocity by 100
                # with nothing downstream to catch it.
                _v = self.venc_override
                if _v <= 0:
                    messagebox.showerror("Invalid VENC",
                                         "VENC must be greater than zero.")
                    return
                if _v > 10.0:
                    if not messagebox.askyesno(
                            "Check the VENC units",
                            f"You entered {_v:g} m/s = {_v*100:.0f} cm/s.\n\n"
                            "This field expects METRES per second. Did you mean "
                            f"{_v/100:g} m/s ({_v:g} cm/s)?\n\n"
                            "Choose No to go back and correct it, or Yes to "
                            f"proceed with {_v:g} m/s."):
                        return
            else:
                self.venc_override = None
            _ptxt = perm_var.get().strip()
            _stxt = signs_var.get().strip()
            try:
                self.direction_permute = (_parse_permute_arg(_ptxt)
                                          if _ptxt else None)
            except Exception as _pe:
                messagebox.showerror("Invalid axis permutation", str(_pe))
                return
            try:
                self.direction_signs = (_parse_signs_arg(_stxt)
                                        if _stxt else None)
            except Exception as _se:
                messagebox.showerror("Invalid axis signs", str(_se))
                return
            self.sim_mode    = mode_var.get()
            self.flow_pattern = pattern_var.get()
            self.run_dicom   = bool(self.dicom_dir)
            confirmed[0]     = True
            root.destroy()

        def _cancel():
            root.destroy()

        tk.Button(footer, text="▶  Run Simulation",
                  bg="#1A3A5C", fg="white",
                  font=("Segoe UI", 10, "bold"),
                  width=22, pady=4,
                  command=_confirm).pack(side="right", padx=(6, 0))
        tk.Button(footer, text="Cancel", width=9,
                  command=_cancel).pack(side="right")

        # ── Size the window to the content, capped to the visible screen ──
        root.update_idletasks()
        _need_w = body.winfo_reqwidth() + _vbar.winfo_reqwidth() + 4
        _need_h = (hdr.winfo_reqheight() + body.winfo_reqheight()
                   + footer.winfo_reqheight() + 8)
        _max_w = int(root.winfo_screenwidth() * 0.95)
        _max_h = int(root.winfo_screenheight() * 0.88)
        root.geometry(f"{min(_need_w, _max_w)}x{min(_need_h, _max_h)}+40+20")
        root.minsize(560, 320)
        _on_body_config()

        root.mainloop()
        return confirmed[0]


# ═════════════════════════════════════════════════════════════════════════════
# GRADIENT FIELD PARSER
# ═════════════════════════════════════════════════════════════════════════════

class GradientFieldParser:
    """
    Parses Siemens ASCII gradient coefficient files (.grad / .txt).

    File format (L4 normalisation, lnorm = 4):
        · Header: CoilName, R0 in metres, gradient amplitude, lnorm
        · Coefficient block: <index> A/B(n, m) <value> <axis>
        · A(n,m) → cosine harmonic, B(n,m) → sine harmonic
        · Primary linear terms A(1,0), A(1,1), B(1,1) are fixed / excluded

    Supported coil files :
        · SC72CD — N_max = 25, 109 coefficients per axis
        · Skyra — N_max = 11, 27 coefficients per axis
        · IMPULSE — N_max = 15, 67 coefficients per axis
    """

    # Compiled regex for coefficient lines: idx A/B(n, m) value axis
    _COEFF_RE = re.compile(
        r"^\s*\d+\s+([AB])\(\s*(\d+)\s*,\s*(\d+)\s*\)"
        r"\s+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s+([xyz])\s*$",
        re.MULTILINE,
    )
    _R0_RE    = re.compile(r"([\d.]+)\s*m\s*=\s*R0", re.IGNORECASE)
    _NAME_RE  = re.compile(r"CoilName\s*=\s*(.+)", re.IGNORECASE)
    _LNORM_RE = re.compile(r"lnorm\s*=\s*(\d+)", re.IGNORECASE)

    # The primary (ideal, linear) term of each gradient coil. Siemens files
    # normally LIST this term with a value of 1.0, because the coefficient set
    # is normalised so that the primary term carries the nominal gradient and
    # every other coefficient is a relative perturbation of it.
    #
    # GradientFieldEvaluator._bz_field already adds the ideal linear term
    # analytically (Bz = X/R0, Y/R0 or Z/R0). If the primary coefficient were
    # also kept in the perturbation sum it would be applied a SECOND time, so
    # Lambda_ii would come out at 2.0 instead of 1.0 and every corrected
    # velocity would be halved -- silently, because the pipeline runs to
    # completion and produces normal-looking figures. The terms are therefore
    # removed here, and what was removed is reported.
    #
    # Only the primary SELF-term of each axis is dropped. Other n = 1 entries
    # (e.g. A(1,0) on the x coil) are genuine linear cross-terms and are kept.
    _PRIMARY_TERM = {"x": ("A", 1, 1), "y": ("B", 1, 1), "z": ("A", 1, 0)}

    @classmethod
    def parse(cls, filepath: str) -> GradientCoefficients:
        """Parse a gradient coefficient file and return a GradientCoefficients object."""
        fp = Path(filepath)
        if not fp.is_file():
            raise FileNotFoundError(f"Gradient file not found: {filepath}")

        with open(fp, "r", errors="ignore") as fh:
            text = fh.read()

        # ── Reference radius ──────────────────────────────────────────────
        m_r0 = cls._R0_RE.search(text)
        R0   = float(m_r0.group(1)) if m_r0 else 0.25  # default 250 mm

        # ── Coil name ─────────────────────────────────────────────────────
        m_nm    = cls._NAME_RE.search(text)
        coilname = m_nm.group(1).strip() if m_nm else fp.stem

        # ── lnorm (normalisation convention, must be 4 for Schmidt) ──────
        m_ln  = cls._LNORM_RE.search(text)
        lnorm = int(m_ln.group(1)) if m_ln else 4   # default 4 if not stated

        # ── Coefficients ──────────────────────────────────────────────────
        coeffs: Dict[str, list] = {"x": [], "y": [], "z": []}
        max_n   = 0
        dropped: Dict[str, float] = {}
        for hit in cls._COEFF_RE.finditer(text):
            typ, n, m_, val, ax = (
                hit.group(1),
                int(hit.group(2)),
                int(hit.group(3)),
                float(hit.group(4)),
                hit.group(5),
            )
            # Drop the primary linear self-term (see _PRIMARY_TERM).
            if cls._PRIMARY_TERM.get(ax) == (typ, n, m_):
                dropped[ax] = val
                continue
            coeffs[ax].append((typ, n, m_, val))
            if n > max_n:
                max_n = n

        if dropped:
            desc = "  ".join(
                f"{ax}:{cls._PRIMARY_TERM[ax][0]}"
                f"({cls._PRIMARY_TERM[ax][1]},{cls._PRIMARY_TERM[ax][2]})={v:.6g}"
                for ax, v in sorted(dropped.items()))
            print("  [parser] Primary linear terms removed from the "
                  f"perturbation sum (added analytically instead):  {desc}")
            off = {a: v for a, v in dropped.items() if abs(v - 1.0) > 1e-3}
            if off:
                print(f"  [parser] WARNING: primary term(s) {off} differ from "
                      "1.0. This file may use a normalisation this parser "
                      "does not implement — verify Lambda(0,0,0) = I before "
                      "trusting any result.")
        else:
            print("  [parser] No primary linear term found in the file "
                  "(already excluded by the vendor) — nothing removed.")

        total = sum(len(v) for v in coeffs.values())
        print(f"  [parser] Coil: {coilname!r}  |  R0 = {R0*1000:.0f} mm  "
              f"|  N_max = {max_n}  |  lnorm = {lnorm}  |  coefficients: "
              f"x={len(coeffs['x'])}  y={len(coeffs['y'])}  z={len(coeffs['z'])}  "
              f"total={total}")

        return GradientCoefficients(
            filepath=str(fp),
            R0=R0,
            coeffs=coeffs,
            coil_name=coilname,
            max_degree=max_n,
            lnorm=lnorm,
        )


# ═════════════════════════════════════════════════════════════════════════════
# SPHERICAL HARMONIC GRADIENT FIELD EVALUATOR
# ═════════════════════════════════════════════════════════════════════════════

class GradientFieldEvaluator:
    """
    Evaluates the spherical harmonic gradient field expansion
    and constructs the voxel-wise 3×3 distortion matrix Λ(r) .

    The field for gradient axis a at position r = (r, θ, φ) is:

        G_a(r, θ, φ) = Σ_{n,m} c_{nm}^a · (r/R₀)^n · Y_n^m(θ, φ)

    where Y_n^m are Schmidt quasi-normalised real spherical harmonics.
    """

    def __init__(self, grad_coeffs: GradientCoefficients):
        self.gc   = grad_coeffs
        self._snorm_cache: Dict[Tuple[int,int], float] = {}
        # CS phase flag (set via set_cs_phase_omit())
        self._cs_phase_omit: bool = False
        # Guard: this evaluator implements Schmidt quasi-normalisation (lnorm=4).
        # Siemens files always use lnorm=4; raise early if a different convention
        # is detected so the user is not silently handed wrong field values.
        if grad_coeffs.lnorm != 4:
            raise ValueError(
                "GradientFieldEvaluator expects lnorm=4 (Schmidt quasi-norm), "
                f"but the coefficient file reports lnorm={grad_coeffs.lnorm}. "
                "Please implement the matching normalisation before proceeding."
            )

    # Allow main() to forward the cfg.cs_phase_omit flag
    def set_cs_phase_omit(self, omit: bool) -> None:
        """Set the Condon–Shortley phase convention.  See _bz_field docstring."""
        self._cs_phase_omit = bool(omit)
        if omit:
            print("  [SH] Condon–Shortley phase STRIPPED for odd-m terms "
                  "(cfg.cs_phase_omit=True).")

    # Allow main() to forward the cfg.sh_normalization flag
    def set_sh_normalization(self, mode: str) -> None:
        """
        Select the m>0 normalisation: "schmidt" (default) or "gradunwarp".

        The two are identical for m = 0 — the entire z-coil therefore agrees
        with the gradunwarp reference to machine precision under either
        setting — and differ by exactly sqrt(2n+1)/2 per m>0 term (verified
        per-term in validate_lambda_vs_gradunwarp.py). See the
        cfg.sh_normalization comment for the self-consistency argument and
        the flux-conservation protocol that discriminates the two on
        acquired data.
        """
        mode = str(mode).strip().lower()
        if mode not in ("schmidt", "gradunwarp"):
            raise ValueError(
                f"sh_normalization must be 'schmidt' or 'gradunwarp', "
                f"got {mode!r}")
        if getattr(self, "_sh_normalization", "schmidt") != mode:
            self._snorm_cache.clear()
        self._sh_normalization = mode
        if mode != "schmidt":
            print("  [SH] m>0 normalisation set to 'gradunwarp' "
                  "(sqrt(2n+1)/2 x Schmidt per term).")

    # Compute the boolean mask marking voxels INSIDE the
    # calibration sphere r ≤ R0. Quantitative results should be reported
    # over this mask only; voxels outside R0 are extrapolated and not
    # physically reliable.
    def inside_r0_mask(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        Z: np.ndarray,
        margin: float = 0.0,
    ) -> np.ndarray:
        """
        Boolean mask of voxels where Lambda(r) is physically meaningful.

        Two distinct effects make voxels outside the calibration sphere
        unusable, and the second is easy to miss:

        1. r > R0 -- the expansion is extrapolated. _bz_field clamps the
           radial factor at min(r/R0, 1), so beyond R0 the perturbation
           terms lose their RADIAL dependence and the radial part of the
           numerical derivative is zero.

           NOTE -- an earlier version of this docstring claimed that Lambda
           therefore "collapses to the identity" outside R0. That is WRONG
           and the correction matters, because it understates the problem.
           The clamp removes only the radial derivative; the ANGULAR part of
           each term, P_n(cos theta) * cos/sin(m phi), still varies with
           position, so d/dx of it is non-zero. Measured on a 48^3 grid (the
           earlier default) with the SC72 coefficients, outside R0:

               median |Lambda - I| = 0.29,  max = 0.43

           So the shell beyond R0 does not look like a perfect linear
           gradient -- it holds an ACTIVELY WRONG field, frozen at the
           r = R0 radial amplitude but still rotating with angle. Either way
           it must be masked; the reason is that the values are meaningless,
           not that they are benign. (Markl et al. 2003 report deviations
           GROWING with distance from isocentre; nothing in this shell can be
           used to support or contradict that.)

        2. |r - R0| < eps -- the central difference straddles the clamp, so
           one stencil point is clamped and the other is not. The resulting
           derivative is a hybrid of both regimes and is meaningless. Pass
           `margin = eps` (the same eps used for compute_lambda) to drop
           this shell.

        Parameters
        ----------
        margin : float
            Shrink the sphere by this radius [m]. Use the finite-difference
            step to exclude the corrupted boundary shell.
        """
        return np.sqrt(X**2 + Y**2 + Z**2) <= (self.gc.R0 - margin)

    # ── Schmidt quasi-normalisation ───────────────────────────────────────

    def _snorm(self, n: int, m: int) -> float:
        key = (n, m)
        if key not in self._snorm_cache:
            delta = 1.0 if m == 0 else 0.0
            val = np.sqrt(
                (2.0 - delta)
                * sp_factorial(n - m, exact=False)
                / sp_factorial(n + m, exact=False)
            )
            # Optional gradunwarp-style normalisation: identical for m = 0,
            # sqrt(2n+1)/2 larger for every m > 0 term (see
            # set_sh_normalization / cfg.sh_normalization).
            if (getattr(self, "_sh_normalization", "schmidt")
                    == "gradunwarp" and m > 0):
                val *= np.sqrt(2.0 * n + 1.0) / 2.0
            self._snorm_cache[key] = val
        return self._snorm_cache[key]

    # ── Bz field for one gradient coil ───────────────────────────────────

    def _bz_field(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        Z: np.ndarray,
        coil_axis: str,
    ) -> np.ndarray:
        """
        Evaluate normalised Bz (dimensionless) for a single gradient coil.
        Includes ideal linear term + all spherical harmonic perturbations.

        r/R0 hard clamp
        ---------------------------
        The factor r_n = min(r/R0, 1) prevents the SH expansion from
        diverging outside the manufacturer's calibration sphere
        (terms grow as r^N for high degree N). Voxels with r > R0 will
        evaluate at the boundary value r = R0 — these are extrapolated
        and NOT physically reliable. In the default ±220 mm simulation FOV
        with the SC72CD R0 = 250 mm, FOV corners reach r ≈ 380 mm, well outside
        R0. Quantitative results from those voxels should be masked
        (see compute_lambda docstring for a helper).

        Condon–Shortley phase convention
        --------------------------------------------
        scipy.special.lpmv(m, n, x) returns P_n^m(x) WITH the
        Condon–Shortley phase (-1)^m. Some Siemens internal
        documentation defines coefficients without this phase. If the
        config flag cs_phase_omit is set, the phase is stripped for
        odd m (terms with even m are unaffected). Default behaviour
        (cs_phase_omit = False) matches the SciPy convention.
        """
        R0   = self.gc.R0
        coef = self.gc.coeffs[coil_axis]
        # read CS phase flag (None when no cfg provided)
        cs_omit = getattr(self, "_cs_phase_omit", False)

        r     = np.sqrt(X**2 + Y**2 + Z**2)
        r_s   = np.where(r < 1e-12, 1e-12, r)
        cos_t = np.clip(np.where(r < 1e-12, 1.0, Z / r_s), -1.0, 1.0)
        phi   = np.arctan2(Y, X)
        r_n   = np.minimum(r / R0, 1.0)   # hard clamp to calibration sphere

        # Ideal linear term
        if coil_axis == "z":
            Bz = Z / R0
        elif coil_axis == "x":
            Bz = X / R0
        else:
            Bz = Y / R0

        # Nonlinear spherical harmonic terms (Schmidt quasi-normalised)
        for (typ, n, m, val) in coef:
            N_nm = self._snorm(n, m)
            P    = lpmv(m, n, cos_t)             # scipy: lpmv(m, n, x)
            # Optionally strip Condon–Shortley phase
            if cs_omit and (m % 2 == 1):
                P = -P
            rn   = r_n ** n
            trig = np.cos(m * phi) if typ == "A" else np.sin(m * phi)
            Bz  += val * rn * N_nm * P * trig

        return Bz

    # ── Λ matrix via central differences ─────────────────────────────────

    def compute_lambda(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        Z: np.ndarray,
        eps: float = 5e-4,
    ) -> np.ndarray:
        """
        Compute 3×3 distortion matrix Λ(r) at every grid point .

        Λ_ji(r) = ∂Bz_i/∂j × R0 (dimensionless)

        POSITION CONVENTION — an assumption worth stating explicitly.
        Markl et al. (2003) note under their Eq. [2] that r = (x,y,z)ᵀ must
        be the TRUE, already spatially unwarped position. This toolbox takes
        the voxel coordinates as given (from the acquisition geometry for
        DICOM, from the grid for simulation) and does NOT unwarp them.

        That is normally correct, because vendors apply geometric distortion
        correction during reconstruction — Lorenz et al. (2014) likewise
        evaluate the field deviations "after 3D distortion correction
        (calculated by the scanner software)". But it is an assumption, and
        it fails in two cases worth checking for your data:
          - distortion correction was disabled at the console;
          - only 2D (in-plane) unwarping was applied, which is common.
            Markl et al. explicitly corrected the slice (z) position
            separately for this reason.
        With uncorrected positions, Λ is evaluated at the wrong point and
        the correction is applied to the wrong voxel.

        Returns
        -------
        Lambda : ndarray, shape (3, 3, *X.shape)
            rows j ∈ {x,y,z}, cols i ∈ {x,y,z}
        """
        shape  = X.shape
        Lambda = np.zeros((3, 3) + shape, dtype=np.float64)
        R0     = self.gc.R0

        for i, ax in enumerate(["x", "y", "z"]):
            # Six shifted evaluations for central-difference gradient
            offsets = [
                (eps, 0,   0),
                (-eps, 0,  0),
                (0,  eps,  0),
                (0, -eps,  0),
                (0,   0,  eps),
                (0,   0, -eps),
            ]
            vals = [
                self._bz_field(X + dx, Y + dy, Z + dz, ax)
                for dx, dy, dz in offsets
            ]
            Lambda[0, i] = (vals[0] - vals[1]) / (2 * eps) * R0
            Lambda[1, i] = (vals[2] - vals[3]) / (2 * eps) * R0
            Lambda[2, i] = (vals[4] - vals[5]) / (2 * eps) * R0

            diag = Lambda[i, i]
            print(f"    [Λ] coil-{ax}  Λ_{ax}{ax} ∈ "
                  f"[{diag.min():.4f}, {diag.max():.4f}]")

        return Lambda

    # ── Derived distortion metrics ────────────────────────────────────────

    @staticmethod
    def strength_deviation(Lambda: np.ndarray) -> np.ndarray:
        """d_i(r) = ||Λ[:,i]||  — relative strength factor, shape (3,*sp)."""
        return np.sqrt(np.sum(Lambda**2, axis=0))   # (3, *spatial)

    @staticmethod
    def angular_deviation_deg(Lambda: np.ndarray) -> np.ndarray:
        """θ_i(r) in degrees — angle between actual and nominal encoding."""
        theta = np.zeros((3,) + Lambda.shape[2:])
        for i in range(3):
            cross_sq = sum(Lambda[j, i]**2 for j in range(3) if j != i)
            theta[i] = np.degrees(
                np.arctan2(np.sqrt(cross_sq), np.abs(Lambda[i, i]))
            )
        return theta

    @staticmethod
    def condition_number_field(Lambda: np.ndarray) -> np.ndarray:
        """
        Voxel-wise 2-norm condition number κ(Λ(r)) = ‖Λ‖₂ · ‖Λ⁻¹‖₂.

        Numerical interpretation
        ------------------------
        κ measures how much input perturbations are amplified by the
        inversion v_true = Λ⁻¹ · v_meas:

            ‖Δv_true‖ / ‖v_true‖ ≤ κ(Λ) · ‖Δv_meas‖ / ‖v_meas‖

        For each voxel:
          κ ≈ 1 → well-conditioned inversion (near identity)
          κ ≲ 10 → stable correction, errors not amplified
          κ ~ 100 → moderate amplification — examine carefully
          κ ≫ 100 → ill-conditioned — the corrector's σ_min gate applies

        NOTE — κ and the invertibility gate are related but distinct.
        κ = σ_max/σ_min is scale-invariant; the corrector gates on σ_min
        itself (cfg.sigma_min_threshold), because 1/σ_min is the absolute
        worst-case noise amplification. A determinant threshold was used
        historically and is deprecated: det = σ1·σ2·σ3 can remain O(1)
        while σ_min → 0, so it never measured conditioning.

        Returns
        -------
        kappa : np.ndarray, shape (Nx, Ny, Nz)
        """
        spatial = Lambda.shape[2:]
        # Reshape to (N_vox, 3, 3) for batched cond computation
        L_flat = Lambda.reshape(3, 3, -1).transpose(2, 0, 1)
        # numpy.linalg.cond is vectorised over the leading axis
        kappa = np.linalg.cond(L_flat)
        # Numerical safety: replace +inf and NaN with a large finite cap
        kappa = np.where(np.isfinite(kappa), kappa, 1e12)
        return kappa.reshape(spatial)


# ═════════════════════════════════════════════════════════════════════════════
# 4D SIMULATION ENGINE
# ═════════════════════════════════════════════════════════════════════════════

class SimulationEngine4D:
    """
    Generates synthetic 4D (3D + time) ground-truth velocity fields,
    applies the forward distortion model, injects noise, and provides
    the corrected field for validation.

    Velocity array convention: shape (3, Nx, Ny, Nz, Nt)
    """

    GAMMA = 2.6752e8   # Gyromagnetic ratio 1H [rad/(s·T)]

    def __init__(self, cfg: SimulationConfig):
        self.cfg = cfg

    # ─── Cardiac waveform ────────────────────────────────────────────────

    def cardiac_waveform(self) -> np.ndarray:
        """
        Physiological cardiac flow waveform discretised into Nt phases.
        Inspired by Markl et al. (2012) aortic waveform;
        peak systolic at phase ~4, diastolic baseline 0.15.

        Waveform composition honesty
        ----------------------------------------
        The two terms below are combined with np.maximum, so for almost
        all t the larger systolic Gaussian dominates and the smaller
        damped-oscillation term only "wins" in the very early and very
        late part of the cycle where the Gaussian is negligible. The
        result is NOT a true diastolic ripple — earlier docstring text
        describing a "diastolic ripple" was misleading. The actual
        waveform is: 15 % diastolic baseline + Gaussian systolic peak
        (FWHM ≈ 1 rad, peak at t ≈ 1.4 rad), normalised to 1 at peak.
        This is adequate for the simulation's purposes (modulating
        velocity magnitude across the cardiac cycle) but should not be
        compared quantitatively to physiological waveform shapes.
        """
        Nt  = self.cfg.Nt
        t   = np.linspace(0, 2 * np.pi, Nt, endpoint=False)
        systolic_peak = np.exp(-8 * ((t - 1.4)**2))
        damped_osc    = 0.08 * np.exp(-1.5 * t) * np.sin(t + 0.5)
        w  = 0.15 + 0.85 * np.maximum(systolic_peak, damped_osc)
        w  = np.clip(w, 0.0, 1.0)
        return w / w.max()   # normalised to [0, 1]

    # ─── Mode 1: Poiseuille (parabolic) tube flow ─────────────────────────

    def generate_poiseuille_4d(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        Z: np.ndarray,
    ) -> np.ndarray:
        """
        Parabolic tube flow along x-axis.

        v_x(ρ, t) = v_max · (1 − ρ²/r²),  ρ = √(y² + z²)

        Direction of the parameterisation
        ---------------------------------
        v_max is PRESCRIBED (cfg.poiseuille_v_peak_m_s) and the flow rate
        Q = v_mean·A = ½·v_max·π·r² is DERIVED from it — not the other way
        round. Deriving v_max from a physiological Q instead gives
        ~0.35 cm/s through this tube, which is below the PC-MRI velocity
        noise floor σ_v = √2·VENC/(π·SNR) and would make the correction
        untestable in simulation.

        Consequence to state in the thesis: because the tube is enlarged
        to 30 mm radius so that it spans ~13 voxels on the default 97³ grid
        over ±220 mm, the DERIVED Q is not physiological (tens of l/min). The
        quantity being validated here is the VELOCITY field, which is
        acquisition-realistic and comparable to VENC; Q is reported only
        for internal consistency between figures, and the physical pipe
        phantom (~8 mm lumen radius) would be under 4 voxels across.

        x-extent now enforced via cfg.tube_length_m: the
        tube extends only over |x| ≤ L/2 rather than across the full FOV.

        Returns v_true: shape (3, Nx, Ny, Nz, Nt)
        """
        cfg   = self.cfg
        shape = X.shape
        Nt    = cfg.Nt
        r     = cfg.tube_radius_m
        L     = cfg.tube_length_m
        # Peak velocity is set to an acquisition-realistic value (config), NOT
        # derived from Q through the coarse-grid tube — otherwise v_max ~0.35
        # cm/s sits below the PC-MRI velocity-noise floor and the correction is
        # untestable. The flow rate is recomputed to stay consistent with the
        # chosen peak velocity and the (enlarged, resolvable) simulation tube.
        v_max = cfg.poiseuille_v_peak_m_s           # peak Poiseuille velocity [m/s]
        Q     = 0.5 * v_max * np.pi * r**2          # v_mean * area  [m³/s]
        cfg.flow_rate_ml_min = Q * 60.0 * 1e6       # keep reporting self-consistent

        print(f"  [sim] Mode-1 Poiseuille:  r={r*1000:.1f} mm  "
              f"L={L*1000:.0f} mm  v_max={v_max*100:.1f} cm/s  "
              f"(Q={cfg.flow_rate_ml_min:.0f} ml/min for this enlarged sim tube)")

        # Tube mask: cylinder of radius r AND length L along x
        rho      = np.sqrt(Y**2 + Z**2)
        in_radius = rho < r
        in_length = np.abs(X) < (L / 2.0)
        tube     = in_radius & in_length

        # Steady parabolic profile (only inside the radial+length mask)
        v_x_2d = np.where(tube, v_max * (1 - (rho / r)**2), 0.0)

        # Temporal behaviour: steady by default (see cfg.mode1_pulsatile).
        if getattr(cfg, "mode1_pulsatile", False):
            # Legacy: mild cardiac modulation, 85–100 % of v_max.
            wave = 0.85 + 0.15 * self.cardiac_waveform()
            print("  [sim] Mode-1 temporal profile: MILDLY PULSATILE "
                  "(0.85 + 0.15·waveform, legacy --mode1-pulsatile)")
        else:
            wave = np.ones(Nt)
            print("  [sim] Mode-1 temporal profile: STEADY "
                  f"(all {Nt} phases identical)")

        v_true  = np.zeros((3,) + shape + (Nt,))
        for t_idx in range(Nt):
            v_true[0, ..., t_idx] = v_x_2d * wave[t_idx]  # primary x-flow

        return v_true

    # ─── Mode 2: Pulsatile Gaussian vessel (cerebrovascular) ────────────

    def generate_pulsatile_vessel_4d(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        Z: np.ndarray,
    ) -> np.ndarray:
        """
        Gaussian vessel profile with physiological pulsatile waveform
        . Oriented along x-axis, centred at origin.

        Axial mode:
            v_x(ρ, t) = v_peak(t) · exp(−ρ²/(2σ²))

        Helical mode (merged from companion simulation):
            v_x = axial component above
            v_y = −swirl_amp · v_peak(t) · g(ρ) · sin(θ)  [tangential]
            v_z = +swirl_amp · v_peak(t) · g(ρ) · cos(θ)  [tangential]
            where θ = arctan2(Z, Y), swirl_amp = cfg.helical_swirl ≈ 0.15 and

                g(ρ) = (ρ/σ) · exp(−ρ²/(2σ²)) / exp(−1/2)

            is a Lamb–Oseen-like swirl profile: it VANISHES on the axis
            (an earlier version used the axial Gaussian profile directly,
            which keeps a finite tangential speed with an undefined
            direction as ρ→0 — a vortex-line discontinuity). g(ρ) peaks at
            ρ = σ with value 1, so cfg.helical_swirl keeps its meaning as
            the peak tangential speed as a fraction of the peak axial
            speed.

            The tangential unit vector (−sin θ, cos θ) in the y-z plane is
            perpendicular to both the vessel axis and the radial direction,
            producing physiological swirling motion (Markl et al. 2012).

        Returns v_true: shape (3, Nx, Ny, Nz, Nt)
        """
        cfg   = self.cfg
        shape = X.shape
        Nt    = cfg.Nt
        fwhm  = cfg.vessel_fwhm_m
        v_pk  = cfg.v_peak_systole
        pat   = cfg.flow_pattern

        sigma   = fwhm / (2 * np.sqrt(2 * np.log(2)))
        rho     = np.sqrt(Y**2 + Z**2)
        profile = np.exp(-rho**2 / (2 * sigma**2))
        wave    = self.cardiac_waveform() * v_pk

        print(f"  [sim] Mode-2 Pulsatile vessel:  FWHM={fwhm*1000:.1f} mm  "
              f"v_peak_systole={v_pk:.2f} m/s  Nt={Nt} phases  "
              f"flow_pattern={pat!r}")

        # ── Resolution check (own >=8-voxel warning heuristic; stricter
        #    literature criterion: pixel/diameter <= 10%, i.e. >=10 voxels
        #    across — Fukuyama et al. 2017, restated by Watanabe et al.
        #    2019) ──────────────────────────────────────────────────────────
        # A vessel spanning only a couple of voxels is dominated by partial
        # volume: the peak velocity is under-read and the vessel mask keeps
        # only a near-uniform plateau, which also makes spatial correlation
        # statistics degenerate. Warn instead of failing, since the grid is
        # the user's choice, but make the limitation visible in the log.
        _vox_mm = float(self.cfg.r_max_m * 2000.0 / max(self.cfg.grid_N - 1, 1))
        _vpd    = (fwhm * 1000.0) / _vox_mm if _vox_mm > 0 else 0.0
        print(f"        Resolution: voxel = {_vox_mm:.2f} mm  ->  "
              f"{_vpd:.1f} voxels across the vessel FWHM")
        if _vpd < 8.0:
            print(f"  [WARN] Only {_vpd:.1f} voxels span the vessel — partial "
                  "volume will dominate.")
            print("         Peak velocity is under-read and the vessel mask "
                  "keeps a near-uniform")
            print("         plateau (spatial correlations there become "
                  "undefined). For a resolved")
            _need_N = int(np.ceil(8.0 * self.cfg.r_max_m * 2000.0 / (fwhm * 1000.0))) + 1
            print(f"         profile use cfg.grid_N >= {_need_N} at the "
                  f"current r_max_m = {self.cfg.r_max_m:.3f} m,")
            print("         or reduce cfg.r_max_m (<= R0/sqrt(3) = 0.144 m "
                  "also removes extrapolation).")

        v_true = np.zeros((3,) + shape + (Nt,))

        if pat == "helical":
            # Tangential unit vector components in y-z plane
            # Safe division: avoid divide-by-zero at vessel centre
            rho_safe  = np.where(rho < 1e-9, 1e-9, rho)
            sin_theta = Z / rho_safe    # sin(arctan2(Z,Y))
            cos_theta = Y / rho_safe    # cos(arctan2(Z,Y))
            swirl     = cfg.helical_swirl
            # Swirl magnitude ∝ ρ·e^(−ρ²/2σ²): zero on the axis, maximum 1
            # at ρ = σ (normalised by σ·e^{−1/2}). See the docstring —
            # replaces the earlier profile-only form whose amplitude stayed
            # finite as ρ→0 while its direction became undefined.
            swirl_profile = (rho / sigma) * profile / np.exp(-0.5)

            for t_idx in range(Nt):
                amp = wave[t_idx]
                v_true[0, ..., t_idx] = profile * amp               # axial
                v_true[1, ..., t_idx] = -swirl * amp * swirl_profile * sin_theta  # tangential y
                v_true[2, ..., t_idx] = +swirl * amp * swirl_profile * cos_theta  # tangential z

            print(f"    Helical swirl amplitude: {swirl*100:.0f}% of peak velocity")

        else:
            # Axial-only: purely axial flow; transverse components are zero.
            # A real vessel may have secondary flow, but for a controlled
            # simulation baseline we enforce v_y = v_z = 0 so that
            # cross-component metrics are not polluted by arbitrary constants.
            for t_idx in range(Nt):
                v_true[0, ..., t_idx] = profile * wave[t_idx]
                # v_true[1] and v_true[2] remain zero (initialised above)

        return v_true

    # ─── Forward distortion ─────────────────

    @staticmethod
    def apply_forward_distortion(
        v_true: np.ndarray,
        Lambda: np.ndarray,
    ) -> np.ndarray:
        """
        Forward gradient-nonlinearity distortion (Markl 2003,
        identical convention to the 3D reference simulation):

            ΔΦ_i / (γ M1_ideal) = Σ_j Λ_ji · v_j_true
                               = (Λᵀ · v_true)_i

        With γ M1_ideal = 1 in this dimensionless framework, the standard-
        reconstruction estimate of the velocity equals the encoded phase
        difference, hence v_measured = Λᵀ · v_true.

Connection to Markl et al. (2003)
        ---------------------------------
        Markl et al. 2003, Eq. [4] introduces the position-dependent first
        gradient moment M1(r) = Λ(r)·M1,ideal. Under gradient nonlinearity
        the phase–velocity relation becomes φ(r) = γ·M1(r)·v(r), so in
        vector form (their Eq. [5])
            Δφ(r) = γ·[Λ(r)·ΔM1,ideal]ᵀ·v(r),
        i.e. v_measured = Λᵀ·v_true once γ·ΔM1,ideal is normalised to 1.
        The matrix Λ(r) used here is exactly their relative field deviation
        Λ_ji = G_ji,model/G_i,ideal (Markl 2003, Eqs. [1]–[2]).

        The TRANSPOSE matters. Λ is not symmetric — the cross-terms satisfy
        Λ_xy ≠ Λ_yx — so using Λ in place of Λᵀ silently mixes the velocity
        components. VelocityCorrector.correct_4d inverts with the matching
        convention; the two are verified against each other by self-test T2.

        This identification also justifies the spatially varying VENC used
        by plot_spatially_varying_venc(),
            VENC(r) = VENC_nominal / ‖Λ(:,i)(r)‖,
        which is the reciprocal of the relative encoding strength
        d_i(r) = ‖Λ(:,i)‖ of Markl 2003, Eq. [11].

        Parameters
        ----------
        v_true : (3, *spatial, Nt)
        Lambda : (3, 3, *spatial)
            Lambda[j, i] = ∂Bz_i / ∂x_j × R0 (rows j, cols i)

        Returns
        -------
        v_measured : same shape as v_true
        """
        Nt = v_true.shape[-1]
        v_meas = np.zeros_like(v_true)
        for t in range(Nt):
            # einsum: v_meas_i = Σ_j Λ_ji · v_j (transpose convention,
            # matches 3D reference and Markl 2003).
            v_meas[..., t] = np.einsum(
                "ji...,j...->i...", Lambda, v_true[..., t]
            )
        return v_meas

    # ─── Noise injection (SNR-based) ──────────────────────────────────────

    def add_phase_noise(
        self,
        v_measured: np.ndarray,
        VENC: float,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """
        Add realistic Gaussian noise to the velocity field, modelling
        phase noise in the PC-MRI phase-difference images.

        NOTE — naming clarification
        ---------------------------
        Phase-contrast velocity maps are derived from PHASE images, not
        magnitude images. Rician noise applies to MR magnitude images;
        the corresponding noise on the phase difference — and therefore on
        the reconstructed velocity — is approximately Gaussian (zero-mean)
        for SNR > 3.

        Noise standard deviation for phase-contrast velocity:

            σ_φ   ≈ 1 / SNR_image          [rad, per image]
            σ_Δφ  = √2 · σ_φ                [rad, phase DIFFERENCE]
            σ_vel = (VENC / π) · σ_Δφ
                  = √2 · VENC / (π · SNR_lin)          [m/s]

        The √2 is not optional. Velocity comes from the phase DIFFERENCE of
        two acquisitions with independent noise, so the two variances add:
        σ_Δφ = √(σ_φ² + σ_φ²) = √2 σ_φ, where σ_φ ≈ 1/SNR is the phase noise
        of a single image; the √2 comes from the subtraction. SNR is defined
        against the magnitude-image noise σ_S.

        Assumes two-point / symmetric (referenced) encoding per direction.
        A balanced four-point scheme distributes noise differently and gives
        a somewhat lower per-component σ_v.

        At SNR = 30 dB (SNR_lin ≈ 31.6) and VENC = 0.9 m/s:
            σ_vel ≈ 12.8 mm/s — realistic for phantom conditions.

        If rng is None, the global numpy RNG is used (legacy
        behaviour, depends on prior np.random.seed calls). Pass an
        explicit np.random.Generator for deterministic isolation across
        runs (e.g., when running Mode 1 + Mode 2 in the same execution).
        """
        snr_dB  = self.cfg.snr_db
        snr_lin = 10 ** (snr_dB / 20.0)
        # PC-MRI phase-difference noise formula (see docstring above).
        # Uses the VENC *argument* — previously this silently read
        # self.cfg.VENC and ignored the parameter, which was harmless only
        # because the sole call site passed cfg.VENC anyway.
        sigma_v = np.sqrt(2.0) * float(VENC) / (np.pi * snr_lin)

        print(f"  [noise] SNR = {snr_dB} dB  →  σ_vel = {sigma_v*100:.2f} cm/s"
              "  (Gaussian phase noise, √2·VENC/(π·SNR))")
        if rng is None:
            noise = np.random.normal(0.0, sigma_v, v_measured.shape)
        else:
            noise = rng.normal(0.0, sigma_v, v_measured.shape)
        # Add in place into the noise buffer instead of allocating a third
        # full-size array (3×N³×Nt float64 = 355 MB at the default grid).
        # IEEE addition is commutative, so the result is bit-identical to
        # v_measured + noise; v_measured itself is not modified.
        noise += v_measured
        return noise



# ═════════════════════════════════════════════════════════════════════════════
# DICOM LOADER
# ═════════════════════════════════════════════════════════════════════════════

class DicomLoader:
    """
    Loads 4D flow DICOM series and converts phase images to velocity volumes.

    Handles TWO formats automatically:
    ──────────────────────────────────────────────────────────────────────────
    Format A — Siemens Enhanced / Multiframe 4D Flow (your Circle of Willis)
        4 files: ww25_Magnitude_venc0_9.dcm
                  ww25_Phases_venc0_9_X.dcm
                  ww25_Phases_venc0_9_Y.dcm
                  ww25_Phases_venc0_9_Z.dcm
        Each file stores ALL slices × ALL cardiac phases as a single 3-D
        pixel array with shape (Nframes, Nrows, Ncols) where
        Nframes = Nslices × Nt. Frame ordering is encoded in
        PerFrameFunctionalGroupsSequence (PFGS).

    Format B — Classic single-slice series (pipe-flow phantom / IMA exports)
        N files, one 2-D image per file, sorted by SliceLocation + TriggerTime.

    VENC extraction (in priority order):
        1. Filename regex venc0_9 → 0.9 m/s ← most reliable for your data
        2. Siemens MrVelocityEncodingMaximumValue (tag 0x0018,0x9214)
        3. Siemens private CSA tag 0x0019,0x1018 (divide by 100 to get m/s)
        4. SequenceName / SeriesDescription text regex
        5. Hard-coded 0.9 m/s fallback with a warning
    ──────────────────────────────────────────────────────────────────────────
    """

    # ── phase → velocity conversion (the ONLY correct mapping) ────────────
    #
    # Siemens phase-contrast DICOM stores the velocity-encoded phase on a
    # 12-bit scale whose calibration is carried by the DICOM rescale
    # attributes:
    #
    #       v = (PV * RescaleSlope + RescaleIntercept) / 4096 * VENC
    #
    # This is an ABSOLUTE calibration: a rescaled value of 0 is exactly zero
    # velocity and ±4096 is exactly ±VENC, identically for all three encoding
    # directions. It must NEVER be replaced by a data-driven min/max
    # normalisation, because that:
    #   (a) gives vx, vy and vz three different, arbitrary scale factors, so
    #       |v| = sqrt(vx²+vy²+vz²) and the velocity vector DIRECTION are both
    #       wrong — the components no longer share a common calibration;
    #   (b) places the zero point at the midpoint of the OBSERVED range rather
    #       than at true zero, adding a constant velocity offset to every
    #       voxel, which makes stationary tissue appear to flow and buries the
    #       real vessels; and
    #   (c) lets a single noise outlier in the background set the scale for the
    #       whole volume.
    PHASE_FULL_SCALE: float = 4096.0

    # Filled in by load_velocity_volume() from the first DICOM header, then
    # compared against the chosen gradient coil file. Class-level so the check
    # can run later in main() without threading it through every return path.
    _scanner_info: dict = {}

    @staticmethod
    def check_coil_matches_scanner(coil_B0: float, scanner_tag: str) -> bool:
        """
        Warn when the gradient coil file does not match the scanner that
        actually acquired the data.

        The coil file is selected by hand in the dialog and B0 is inferred
        from its FILENAME, so nothing otherwise prevents 3 T Skyra
        coefficients being applied to 7 T data. That produces a Lambda that
        is wrong everywhere — smoothly, plausibly wrong, with no visible
        artefact — and every corrected velocity inherits the error. Since the
        DICOM header states the field strength, the two can be compared.

        Returns True when they agree (or cannot be checked).
        """
        hdr_b0 = DicomLoader._scanner_info.get("B0_T")
        if hdr_b0 is None:
            return True                       # nothing to compare against
        try:
            hdr_b0 = float(hdr_b0)
        except (TypeError, ValueError):
            return True
        if abs(hdr_b0 - float(coil_B0)) <= 0.6:      # 3T vs 7T is unmissable
            print(f"  [check] Coil file matches the data on field strength: "
                  f"{hdr_b0:.1f} T header, {float(coil_B0):.1f} T coil file.")
            # Field strength alone is NOT sufficient. Two 7 T systems can use
            # entirely different gradient coils (a 200 mm head-only Athena on
            # Numaris X vs a 250 mm whole-body SC72 on VB17), and B0 is
            # identical for both — so this check passes while Lambda is
            # computed from the wrong geometry. The software generation
            # separates them: Athena coefficients are Numaris X ("XA"), while
            # a "syngo MR B17"/VB scanner predates that coil entirely.
            _sw = str(DicomLoader._scanner_info.get("software", "")).lower()
            _coil_lc = str(scanner_tag).lower()
            _is_xa = ("xa" in _sw or "numaris x" in _sw)
            _is_vb = ("b17" in _sw or _sw.startswith("n4")
                      or _sw.startswith("syngo mr b"))
            _wants_xa = ("impulse" in _coil_lc or "terra" in _coil_lc
                         or "athena" in _coil_lc)
            if _wants_xa and _is_vb:
                print("\n  " + "!" * 68)
                print("  [check] COIL FILE IS PROBABLY STILL WRONG "
                      "(field strength alone cannot tell)")
                print(f"          Software  : {_sw}  → a VB-generation scanner")
                print(f"          Coil file : {scanner_tag}  → Numaris X coil")
                print("          The Athena/IMPULSE head coil belongs to a "
                      "Numaris X system.")
                print("          VB17-era 7 T data normally needs the SC72 "
                      "whole-body file.")
                print("          Verify the serial number against the scanner "
                      "before trusting")
                print("          any result from this run.")
                print("  " + "!" * 68 + "\n")
                return False
            if _wants_xa and not _is_xa and _sw:
                print(f"  [check] NOTE: software '{_sw}' does not look like "
                      "Numaris X, but the coil\n          file is a Numaris X "
                      "coil. Confirm the coil file is right for this scanner.")
            print(f"          Serial {DicomLoader._scanner_info.get('serial','?')}"
                  f"  ·  {DicomLoader._scanner_info.get('model','?')} ✓")
            return True
        model = DicomLoader._scanner_info.get("model", "") or "unknown model"
        print("\n  " + "!" * 68)
        print("  [check] GRADIENT COIL FILE DOES NOT MATCH THIS DATA")
        print(f"          DICOM header : {hdr_b0:.1f} T  ({model})")
        print(f"          Coil file    : {float(coil_B0):.1f} T  "
              f"({scanner_tag})")
        print("          Lambda is computed from the coil geometry, so the")
        print("          wrong file gives a wrong correction everywhere —")
        print("          smoothly wrong, with no visible artefact. Select the")
        print("          .grad file for the scanner that acquired this series.")
        print("  " + "!" * 68 + "\n")
        return False

    @staticmethod
    def resolve_phase_full_scale(lo: float, hi: float, slope: float,
                                 inter: float, bits_stored: int = None):
        """
        Determine the phase full-scale FS from the DICOM rescale attributes.

        FS is the rescaled phase value that corresponds to exactly +VENC:

            v = (PV * RescaleSlope + RescaleIntercept) / FS * VENC

        BitsStored must NOT be used on its own for this. BitsStored is the
        depth of the pixel CONTAINER, not the depth of the phase encoding:
        Siemens routinely writes 12-bit phase (FS = 4096) into a 16-bit
        container, so 2^BitsStored over-estimates FS by a factor of 16 and
        every velocity comes out 16x too small — or, worse, the resulting
        FS is so large that the bipolar/unipolar test below misfires and the
        whole volume is shifted by a constant offset of ~VENC per component.

        The calibration actually lives in RescaleIntercept: for the standard
        Siemens bipolar mapping the header centres the data on zero, so
        RescaleIntercept = -FS (e.g. RS = 2, RI = -4096 for 12-bit phase
        stored 0..4095 → rescaled -4096..+4094 → FS = 4096).

        Rule: among the NOMINAL candidates (never a data-driven min/max —
        see the note on PHASE_FULL_SCALE above), take the smallest one that
        still contains the observed rescaled range. That resolves

            |RI| = 4096, BitsStored = 16, data ±4096  → FS = 4096   (12-bit in a 16-bit container)
            |RI| = 32768, BitsStored = 16, data ±32768 → FS = 32768 (true 16-bit phase)
            |RI| = 1024, BitsStored = 10, data ±1024  → FS = 1024   (10-bit phase)
            RI = 0, RS = 1, data 0..4095              → FS = 4096   (unipolar 12-bit)

        Returns (fs, fs_source_string).
        """
        span  = max(abs(float(lo)), abs(float(hi)))
        cands = []
        if float(inter) < -1.0:                     # vendor-supplied zero point
            cands.append((abs(float(inter)), "|RescaleIntercept|"))
        if bits_stored and 6 <= int(bits_stored) <= 16:
            cands.append((float(2 ** int(bits_stored)),
                          f"2^BitsStored(={int(bits_stored)})"))
        cands.append((DicomLoader.PHASE_FULL_SCALE, "12-bit default"))

        fits = [c for c in cands if c[0] >= span * (1.0 - 1e-6)]
        if fits:
            return min(fits, key=lambda c: c[0])
        # Nothing nominal is wide enough (unusual): round the observed span
        # up to the next power of two rather than normalising by the data.
        fs = float(2 ** int(np.ceil(np.log2(max(span, 1.0)))))
        return fs, "next power of two above the observed range"

    @staticmethod
    def phase_to_velocity(raw_px, slope: float, inter: float,
                          venc: float, label: str = "",
                          bits_stored: int = None) -> np.ndarray:
        """
        Convert stored phase pixel values to velocity in m/s.

        v = (PV * RescaleSlope + RescaleIntercept) / FS * VENC

        where FS is the phase full-scale resolved by
        resolve_phase_full_scale() from the DICOM rescale attributes (with
        BitsStored used only as an upper bound, never on its own).

        Parameters
        ----------
        raw_px : array_like   raw stored pixel values (NOT pre-rescaled)
        slope  : float        RescaleSlope   (0028,1053)
        inter  : float        RescaleIntercept (0028,1052)
        venc   : float        VENC in m/s
        label  : str          axis label for the diagnostic printout
        bits_stored : int     BitsStored (0028,0101); used only as one
                              candidate full-scale, bounded by the rescale
                              attributes and the observed range.

        Returns velocity in m/s, same shape as raw_px.
        """
        # MEMORY. Every step below works IN PLACE on one float64 buffer. The
        # earlier form (raw*RS + RI, then px/fs*VENC) held up to four full-size
        # float64 temporaries at once — about 2 GB for one direction of a
        # 260×320×52×17 head series, which exhausted RAM on an 8–16 GB machine.
        # The operations and their order are unchanged ((x*RS)+RI, then
        # (x/fs)*VENC), so the result is bit-identical. np.array(..., copy)
        # guarantees the caller's array is never modified.
        px = np.array(raw_px, dtype=np.float64)
        px *= float(slope)
        px += float(inter)
        lo, hi = float(px.min()), float(px.max())

        # Full-scale from the rescale calibration, NOT from BitsStored alone.
        fs, fs_src = DicomLoader.resolve_phase_full_scale(
            lo, hi, slope, inter, bits_stored)
        if abs(fs - DicomLoader.PHASE_FULL_SCALE) > 1e-6:
            print(f"  [DICOM] axis-{label}: full-scale = {fs:.0f} "
                  f"from {fs_src} (not the 4096 default).")

        # Bipolar iff the header (or the data) already places zero phase at
        # rescaled 0. Both tests are relative to the RESOLVED fs above; with
        # fs taken from BitsStored this comparison silently failed for
        # 12-bit-in-16-bit data (RI = -4096 is not < -0.10 * 65536), which
        # sent correctly calibrated bipolar phase down the unipolar branch
        # and offset every component by ~-VENC.
        if inter < -0.10 * fs or lo < -0.10 * fs:
            # Standard bipolar storage — the header already centres the data
            # on zero, so the textbook formula applies directly. Decide this
            # from RescaleIntercept where possible, NOT from whether the data
            # happens to contain negative values: a volume with flow in only
            # one direction has none, and would be misclassified.
            px  /= fs
            px  *= venc
            vel  = px
            conv = f"(PV*{slope:g}{inter:+g})/{fs:g} * VENC"
        else:
            # Unipolar storage (stored 0..4095 with RS=1, RI=0): the header
            # carries no zero point. Centre on the NOMINAL midpoint of the
            # stored range — never on the observed min/max, which is noise.
            half = fs / 2.0
            px  -= half
            px  /= half
            px  *= venc
            vel  = px
            conv = (f"(PV*{slope:g}{inter:+g}-{half:g})/{half:g} * VENC "
                    "[unipolar fallback]")
            print(f"  [DICOM] NOTE axis-{label}: rescaled phase range "
                  f"[{lo:.0f}, {hi:.0f}] carries no zero point in the header "
                  f"(RI = {inter:g}). Assuming unipolar storage "
                  f"0..{fs-1:.0f} with {half:.0f} = zero velocity. "
                  "VERIFY this against your vendor's convention.")

        print(f"  [DICOM] axis-{label}: RS={slope:g}  RI={inter:g}  "
              f"rescaled range [{lo:.0f}, {hi:.0f}]")
        print(f"           v = {conv}  ->  "
              f"[{vel.min()*100:+.1f}, {vel.max()*100:+.1f}] cm/s "
              f"(VENC = {venc*100:.0f} cm/s)")

        # Sanity check: a correct conversion spans close to ±VENC and is
        # roughly centred on zero over the whole volume (background phase is
        # symmetric noise). Large asymmetry usually means a wrong RS/RI, a
        # wrong full scale, or phase wrap.
        span_frac = max(abs(vel.min()), abs(vel.max())) / max(venc, 1e-9)
        if span_frac > 1.05:
            print(f"           WARNING axis-{label}: |v| reaches "
                  f"{span_frac*100:.0f}% of VENC — check for phase wrap or a "
                  "wrong VENC.")
        elif span_frac < 0.50:
            print(f"           WARNING axis-{label}: |v| only reaches "
                  f"{span_frac*100:.0f}% of VENC — check RescaleSlope / "
                  f"RescaleIntercept and the {fs:g} full scale.")

        # Zero-point check. Over a WHOLE volume the phase must straddle zero:
        # air background is uniformly random phase, and stationary tissue sits
        # at zero. A conversion whose entire volume has one sign and whose
        # midpoint is far from zero is not physically possible — it means the
        # zero point (fs / the bipolar-vs-unipolar decision) is wrong, and it
        # is exactly the failure mode that makes phase data get rejected
        # downstream as "magnitude data" by the strict input check.
        v_lo, v_hi = float(vel.min()), float(vel.max())
        v_mid = 0.5 * (v_lo + v_hi)
        if v_lo * v_hi > 0.0 and abs(v_mid) > 0.30 * venc:
            print(f"           WARNING axis-{label}: every voxel in the volume "
                  f"has the SAME sign and the range is centred on "
                  f"{v_mid*100:+.1f} cm/s, not on 0. Real phase data always "
                  "straddles zero (air background is random phase). The zero "
                  f"point is wrong — check RescaleIntercept ({inter:g}) and "
                  f"the resolved full scale ({fs:g}, {fs_src}).")
        return vel

    # ── public API ────────────────────────────────────────────────────────

    @staticmethod
    def find_dicoms(directory: str) -> List[str]:
        """
        Recursively find DICOM files in directory.

        Accepts .dcm / .ima by extension, and ALSO extension-less files
        (e.g. Siemens 'MR.1.3.12…' exports) by checking the DICOM magic
        bytes ('DICM' at offset 128). Without this, steady-flow series whose
        files carry no extension would be silently missed.
        """
        exts = {".ima", ".dcm"}

        def _is_dicom(path: str) -> bool:
            if Path(path).suffix.lower() in exts:
                return True
            try:                              # check DICOM magic 'DICM' @ 128
                with open(path, "rb") as fh:
                    fh.seek(128)
                    return fh.read(4) == b"DICM"
            except Exception:
                return False

        files = []
        for root, _, fnames in os.walk(directory):
            for fn in fnames:
                p = os.path.join(root, fn)
                if _is_dicom(p):
                    files.append(p)
        files.sort()
        print(f"  [DICOM] Found {len(files)} files in '{directory}'")
        return files

    @staticmethod
    def load_series(files: List[str]) -> Tuple[np.ndarray, dict]:
        """
        Load 4D flow DICOM data and return a 5-D velocity array.

        Parameters
        ----------
        files : list of str
            Paths returned by find_dicoms(). May be 4 multiframe files
            (Format A) or N single-slice files (Format B).

        Returns
        -------
        vel_vol : np.ndarray, shape (3, Nx, Ny, Nz, Nt), dtype float32
            vel_vol[0] = vx, [1] = vy, [2] = vz, in m/s.
            Positive convention: x = L→R, y = A→P, z = F→H (scanner frame).
        hdr : dict
            "VENC" : float, m/s
            "PixelSpacing_mm" : [dy_mm, dx_mm]
            "SliceThickness_mm" : float
            "Nt" : int
            "Nslices" : int
            "format" : "multiframe" | "singleframe"
        """
        if not PYDICOM_AVAILABLE:
            raise ImportError("pydicom required.  pip install pydicom")

        # ── Detect format ─────────────────────────────────────────────────
        ds0 = pydicom.dcmread(files[0], stop_before_pixels=True)

        # Report what scanner actually acquired this data. The coil file is
        # chosen by the operator and the program infers B0 from its FILENAME
        # alone, so a mismatched .grad file (e.g. the 3 T Skyra coefficients
        # applied to 7 T data) would otherwise pass unnoticed and produce a
        # confidently wrong Lambda. Printing the header lets that be caught,
        # and check_coil_matches_scanner() below turns it into a warning.
        DicomLoader._scanner_info = {
            "B0_T":   getattr(ds0, "MagneticFieldStrength", None),
            "model":  str(getattr(ds0, "ManufacturerModelName", "") or ""),
            "maker":  str(getattr(ds0, "Manufacturer", "") or ""),
            "station": str(getattr(ds0, "StationName", "") or ""),
            "serial": str(getattr(ds0, "DeviceSerialNumber", "") or ""),
            "software": str(getattr(ds0, "SoftwareVersions", "") or ""),
            # (0018,5100) — Head-First-Supine etc. The Λ evaluation assumes
            # the DICOM patient (LPS) frame coincides with the gradient-coil
            # frame, which on Siemens holds for HFS; see the frame-of-
            # reference guard where the DICOM grid is built.
            "patient_position": str(getattr(ds0, "PatientPosition", "") or ""),
        }
        _si = DicomLoader._scanner_info
        _b0 = _si["B0_T"]
        print(f"  [DICOM] Scanner from header: "
              f"{_si['maker'] or '?'} {_si['model'] or '?'}"
              + (f"  ·  {float(_b0):.1f} T" if _b0 else "  ·  field strength "
                 "not in header")
              + (f"  ·  station {_si['station']}" if _si["station"] else ""))

        nframes_tag = int(getattr(ds0, "NumberOfFrames", 1))
        is_multiframe = (nframes_tag > 1)

        if is_multiframe:
            return DicomLoader._load_multiframe(files)

        # Try 3-direction-interleaved phase first (Ideal Phantom cardiac
        # cycle: 11 cardiac phases × 3 dirs). Falls back to classic single-
        # component single-frame if the interleaved pattern is not present.
        interleaved = DicomLoader._try_load_interleaved(files)
        if interleaved is not None:
            return interleaved
        return DicomLoader._load_singleframe(files)

    # ── private helpers ───────────────────────────────────────────────────

    @staticmethod
    def _extract_venc(files: List[str], datasets=None) -> float:
        """
        Extract VENC from filename, DICOM header tags, or fallback.

        Returns VENC in m/s.
        """
        venc = None

        # Priority 0 — explicit user override (DICOM_VENC_OVERRIDE_MS at top of
        # file). 4D-flow VENC is often NOT in the header, so this lets you pin
        # the true value for your scan/phantom instead of relying on detection.
        if DICOM_VENC_OVERRIDE_MS is not None:
            venc = float(DICOM_VENC_OVERRIDE_MS)
            print("  [DICOM] VENC set by user override "
                  f"(DICOM_VENC_OVERRIDE_MS): {venc:.2f} m/s "
                  f"({venc*100:.0f} cm/s)")
            return venc

        # Priority 1 — filename regex venc0_9 or venc0.9 or venc90
        for f in files:
            name = Path(f).stem.lower()
            # Matches: venc0_9 → 0.9 venc0.9 → 0.9 venc90 → 0.9
            m = re.search(r"venc(\d+)[_.](\d+)", name)
            if m:
                venc = float(m.group(1)) + float(m.group(2)) / (
                    10 ** len(m.group(2)))
                print("  [DICOM] VENC from filename pattern "
                      f"'venc{m.group(1)}_{m.group(2)}': {venc:.2f} m/s")
                return venc
            # Integer form: venc90 → 90 cm/s → 0.90 m/s
            m2 = re.search(r"venc(\d{2,4})", name)
            if m2:
                raw = float(m2.group(1))
                # Heuristic: if value > 10 assume cm/s
                venc = raw / 100.0 if raw > 10 else raw
                print("  [DICOM] VENC from filename integer "
                      f"'venc{int(raw)}': {venc:.2f} m/s")
                return venc

        # Priority 2 — standard DICOM tag MrVelocityEncodingMaximumValue
        ds_list = datasets or [pydicom.dcmread(f, stop_before_pixels=True)
                                for f in files[:1]]
        ds0 = ds_list[0]
        try:
            val = ds0[0x0018, 0x9214].value   # in mm/s
            venc = float(val) / 1000.0        # mm/s → m/s
            print(f"  [DICOM] VENC from tag (0018,9214): {venc:.2f} m/s")
            return venc
        except (KeyError, AttributeError):
            pass

        # Priority 3 — Siemens CSA private tag 0x0019,0x1018
        # Stored in cm/s in many Siemens product sequences
        _csa_note = None
        try:
            raw = float(ds0[0x0019, 0x1018].value)
            # Guard: the tag in older Siemens SW sometimes stores the raw
            # gradient encode value (≫ 1000), not the cm/s VENC. It can also
            # hold an unrelated CSA field the vendor packed at the same
            # address, which is what happened on a pipe-flow rig at VENC
            # 0.07 m/s = 7 cm/s: the tag read 11700/17600, wildly outside a
            # plausible VENC and INCONSISTENT ACROSS TWO RESOLUTIONS OF THE
            # SAME RIG (VENC does not change with matrix size), which is a
            # second, independent reason not to trust it even before the
            # range guard rejects it.
            # Accept only if 5 ≤ raw ≤ 500 (reasonable VENC range in cm/s).
            if 5.0 <= raw <= 500.0:
                venc = raw / 100.0   # cm/s → m/s
                print("  [DICOM] VENC from CSA 0x0019,0x1018: "
                      f"{venc*100:.0f} cm/s → {venc:.2f} m/s")
                return venc
            else:
                _csa_note = (f"CSA tag 0x0019,0x1018 = {raw:.0f} "
                             "(outside the 5-500 cm/s plausibility range; "
                             "not a VENC value, ignored)")
                print(f"  [DICOM] {_csa_note}")
        except (KeyError, AttributeError):
            pass

        # Priority 4 — text regex in SequenceName / SeriesDescription
        for tag_name in ["SequenceName", "SeriesDescription", "ProtocolName"]:
            val = str(getattr(ds0, tag_name, ""))
            m = re.search(r"v?enc[_\s]*(\d+(?:[._]\d+)?)", val,
                          re.IGNORECASE)
            if m:
                raw_str = m.group(1).replace("_", ".")
                raw = float(raw_str)
                venc = raw / 100.0 if raw > 10 else raw
                print(f"  [DICOM] VENC from {tag_name} text "
                      f"'{m.group(0)}': {venc:.2f} m/s")
                return venc

        # Nothing found. VELOCITIES SCALE LINEARLY WITH VENC, so an invented
        # fallback silently corrupts every quantitative number downstream —
        # this toolbox is run against whatever DICOM folder is pointed at it,
        # not only the protocol it was developed on, so guessing is never
        # safe here. DICOM_REQUIRE_VENC defaults to True for exactly this
        # reason; only set it False if you have separately confirmed a fixed
        # fallback is correct for every dataset this run will ever see.
        detail = ("filename pattern, DICOM tag (0018,9214), and text fields "
                  "SequenceName/SeriesDescription/ProtocolName all failed")
        if _csa_note:
            detail += f"; {_csa_note}"
        if DICOM_REQUIRE_VENC:
            raise ValueError(
                f"VENC could not be determined automatically ({detail}). "
                "Set DICOM_VENC_OVERRIDE_MS to the protocol VENC in m/s "
                "(e.g. 0.07 for a 7 cm/s pipe-flow rig, 0.90 for a typical "
                "Circle-of-Willis scan) — read it from the scanner console, "
                "the protocol PDF, or the operator log; do not guess from a "
                "header field whose meaning has not been confirmed.")
        venc = 0.9
        print(f"  [DICOM] WARNING: VENC could not be determined ({detail}). "
              f"Using fallback {venc:.2f} m/s. THIS FALLBACK IS ONLY CORRECT "
              "FOR A 0.90 m/s PROTOCOL. Verify against your scan and set "
              "DICOM_VENC_OVERRIDE_MS explicitly if it does not match.")
        return venc

    @staticmethod
    def _load_multiframe(files: List[str]) -> Tuple[np.ndarray, dict]:
        """
        Load Siemens Enhanced Multiframe 4D flow DICOM.

        Expected layout (your Circle of Willis data):
            ww25_Magnitude_venc0_9.dcm — not used for velocity
            ww25_Phases_venc0_9_X.dcm — vx, shape pixels (Nframes, Ny, Nx)
            ww25_Phases_venc0_9_Y.dcm — vy
            ww25_Phases_venc0_9_Z.dcm — vz

        Frame order inside each file:
            frames are ordered as slice_0/phase_0, slice_0/phase_1, …,
            slice_0/phase_Nt-1, slice_1/phase_0, … (slice-major)
            OR phase_0/slice_0, phase_0/slice_1, … (phase-major).
            We read the TemporalPositionIdentifier and InStackPositionNumber
            from PerFrameFunctionalGroupsSequence to get the correct order.
        """
        print("  [DICOM] Format: Siemens Enhanced Multiframe 4D Flow")

        # ── Separate magnitude and phase files ─────────────────────────
        mag_files   = []
        phase_files = {"x": None, "y": None, "z": None}

        for f in files:
            name_lc = Path(f).stem.lower()
            # Check for axis label in filename
            if any(k in name_lc for k in ["magnitude", "mag"]):
                mag_files.append(f)
            elif name_lc.endswith("_x") or "_x." in name_lc or "phase_x" in name_lc:
                phase_files["x"] = f
            elif name_lc.endswith("_y") or "_y." in name_lc or "phase_y" in name_lc:
                phase_files["y"] = f
            elif name_lc.endswith("_z") or "_z." in name_lc or "phase_z" in name_lc:
                phase_files["z"] = f
            else:
                # Try ImageType tag or SeriesDescription for axis
                try:
                    ds_tmp = pydicom.dcmread(f, stop_before_pixels=True)
                    itype  = " ".join(str(x) for x in
                                      getattr(ds_tmp, "ImageType", []))
                    sdesc  = str(getattr(ds_tmp, "SeriesDescription", ""))
                    combo  = (itype + " " + sdesc).lower()
                    if "magnitude" in combo or "_mag" in combo:
                        mag_files.append(f)
                    elif combo.endswith(" x") or " x " in combo or "vx" in combo:
                        phase_files["x"] = f
                    elif combo.endswith(" y") or " y " in combo or "vy" in combo:
                        phase_files["y"] = f
                    elif combo.endswith(" z") or " z " in combo or "vz" in combo:
                        phase_files["z"] = f
                    else:
                        mag_files.append(f)   # default to magnitude
                except Exception:
                    mag_files.append(f)

        print("  [DICOM] Identified:  "
              f"magnitude={len(mag_files)}  "
              f"phase_x={'✓' if phase_files['x'] else '✗'}  "
              f"phase_y={'✓' if phase_files['y'] else '✗'}  "
              f"phase_z={'✓' if phase_files['z'] else '✗'}")

        # ── VENC ───────────────────────────────────────────────────────
        venc = DicomLoader._extract_venc(files)

        # ── Load one phase file to get geometry ────────────────────────
        # Use whichever axis file is available
        ref_file = (phase_files["x"] or phase_files["y"]
                    or phase_files["z"])
        if ref_file is None:
            raise ValueError(
                "[DICOM] Could not identify any phase-encoding file. "
                "Filenames must end with _X, _Y, or _Z (case-insensitive).")

        ref_ds   = pydicom.dcmread(ref_file)
        nframes  = int(ref_ds.NumberOfFrames)
        ny, nx   = int(ref_ds.Rows), int(ref_ds.Columns)

        # Extract per-frame slice + temporal indices from PFGS
        nz, nt = DicomLoader._parse_frame_order(ref_ds, nframes)
        print(f"  [DICOM] Grid:  Nx={nx}  Ny={ny}  Nz={nz}  Nt={nt}  "
              f"(total frames = {nz*nt})")

        # ── Resolve the multiframe frame ordering ONCE ──────────────────
        # Files without per-frame geometry are ambiguous (phase-major vs
        # slice-major). Detect from the clean magnitude pixel data when a
        # magnitude file is present, else from the phase reference; a manual
        # override (DICOM_MULTIFRAME_ORDER) takes precedence.
        if DICOM_MULTIFRAME_ORDER in ("phase_major", "slice_major"):
            frame_order = DICOM_MULTIFRAME_ORDER
            print(f"  [DICOM] Frame order: {frame_order}  (manual override)")
        elif not hasattr(ref_ds, "PerFrameFunctionalGroupsSequence"):
            _det_px = None; _src = "phase reference"
            if mag_files:
                try:
                    _det_px = pydicom.dcmread(mag_files[0]).pixel_array
                    _src = "magnitude"
                except Exception:
                    _det_px = None
            if _det_px is None:
                _det_px = ref_ds.pixel_array
            frame_order, _conf = DicomLoader._detect_frame_order(_det_px, nz, nt)
            print(f"  [DICOM] Frame order: {frame_order}  "
                  f"(auto-detected from {_src}, margin={_conf:.3f})")
        else:
            frame_order = "phase_major"   # PFGS present → LUT uses metadata anyway

        px_sp = [float(v) for v in
                 getattr(ref_ds, "PixelSpacing", [1.0, 1.0])]
        sl_th = float(getattr(ref_ds, "SliceThickness", 1.0))

        # ── Load each phase volume and convert to velocity ─────────────
        vel_vol = np.zeros((3, nx, ny, nz, nt), dtype=np.float32)
        axes    = [("x", 0), ("y", 1), ("z", 2)]

        for ax_label, ax_idx in axes:
            f = phase_files[ax_label]
            if f is None:
                print(f"  [DICOM] WARNING: no file for axis {ax_label} — "
                      f"velocity component v{ax_label} will be zero.")
                continue

            ds      = pydicom.dcmread(f)
            # Stored integers, NOT a float64 copy: phase_to_velocity makes the
            # one float64 buffer it needs (saves ~560 MB per direction).
            raw_px  = ds.pixel_array                     # (Nframes, Ny, Nx)
            slope   = float(getattr(ds, "RescaleSlope",     1.0))
            inter   = float(getattr(ds, "RescaleIntercept", 0.0))
            bits    = getattr(ds, "BitsStored", None)
            bits    = int(bits) if bits is not None else None

            # Phase → velocity, using the DICOM rescale calibration:
            #     v = (PV * RS + RI) / 4096 * VENC
            # The phase image encoded along x gives vx, along y gives vy and
            # along z gives vz; because all three share this one absolute
            # calibration, |v| = sqrt(vx²+vy²+vz²) is meaningful.
            _rmin, _rmax = float(raw_px.min()), float(raw_px.max())
            if _rmax - _rmin < 1.0:
                print(f"  [DICOM] WARNING axis-{ax_label}: pixel range too "
                      f"narrow ({_rmin:.1f}–{_rmax:.1f}), "
                      "velocity will be ~0.")
                continue

            vel_frames = DicomLoader.phase_to_velocity(
                raw_px, slope, inter, venc, label=ax_label,
                bits_stored=bits)  # (Nframes,Ny,Nx)

            # ── Unpack frames into (Nx, Ny, Nz, Nt) ──────────────────
            frame_lut = DicomLoader._build_frame_lut(ds, nframes, nz, nt,
                                                     order=frame_order)
            for frame_idx, (s_idx, t_idx) in enumerate(frame_lut):
                if 0 <= s_idx < nz and 0 <= t_idx < nt:
                    # vel_frames[frame_idx] is (Ny, Nx) → transpose to (Nx,Ny)
                    vel_vol[ax_idx, :, :, s_idx, t_idx] = \
                        vel_frames[frame_idx].T

            # Release this direction's velocity buffers before reading the
            # next one. `ds` is deliberately kept: the geometry block after
            # this loop reads SpacingBetweenSlices / ImagePositionPatient /
            # ImageOrientationPatient from the last phase file.
            del raw_px, vel_frames
            print(f"  [DICOM] axis-{ax_label} loaded  "
                  f"peak |v| = {max(abs(float(vel_vol[ax_idx].min())), abs(float(vel_vol[ax_idx].max())))*100:.1f} cm/s")

        # ── Magnitude volume for the signal mask ─────────────────────────
        # Phase-contrast background is random phase noise spanning ±VENC. In a
        # head that background occupies most of the FOV and completely swamps
        # the vessels in fig05 and in the correction statistics. The multiframe
        # loader has already located the magnitude file (for frame-order
        # detection) — unpack it on the SAME frame LUT and keep it so the
        # downstream signal mask can isolate the actual anatomy.
        mag_vol = None
        if mag_files:
            try:
                mds   = pydicom.dcmread(mag_files[0])
                m_px  = mds.pixel_array.astype(np.float32)
                m_lut = DicomLoader._build_frame_lut(
                    mds, int(m_px.shape[0]), nz, nt, order=frame_order)
                m_vol = np.zeros((nx, ny, nz, nt), np.float32)
                for f_i, (s_i, t_i) in enumerate(m_lut):
                    if 0 <= s_i < nz and 0 <= t_i < nt:
                        m_vol[:, :, s_i, t_i] = m_px[f_i].T
                mag_vol = m_vol.mean(axis=-1)          # (Nx,Ny,Nz)
                print("  [DICOM] Magnitude volume unpacked for masking: "
                      f"{'×'.join(map(str, mag_vol.shape))}")
            except Exception as exc:
                print(f"  [DICOM] Magnitude unpack failed ({exc}) — "
                      "no signal mask from the multiframe series.")

        # True slice step for the multiframe export: prefer the explicit
        # SpacingBetweenSlices tag (top-level, or inside the shared
        # functional groups of an enhanced-multiframe file). The caller
        # falls back to SliceThickness when this stays None.
        _sl_step_mm = None
        try:
            if hasattr(ds, "SpacingBetweenSlices"):
                _sl_step_mm = float(ds.SpacingBetweenSlices)
            else:
                _sl_step_mm = float(
                    ds.SharedFunctionalGroupsSequence[0]
                      .PixelMeasuresSequence[0].SpacingBetweenSlices)
        except Exception:
            _sl_step_mm = None

        hdr = {
            "VENC":               venc,
            "PixelSpacing_mm":    px_sp,
            "SliceThickness_mm":  sl_th,
            "SliceSpacing_mm":    _sl_step_mm,
            "Nt":                 nt,
            "Nslices":            nz,
            "format":             "multiframe",
            "magnitude_volume":   mag_vol,
            "ImagePositionPatient":
                ([float(v) for v in ds.ImagePositionPatient]
                 if hasattr(ds, "ImagePositionPatient") else None),
            "ImageOrientationPatient":
                ([float(v) for v in ds.ImageOrientationPatient]
                 if hasattr(ds, "ImageOrientationPatient") else None),
        }
        print("  [DICOM] Velocity volume built: "
              f"shape = {vel_vol.shape}  VENC = {venc*100:.0f} cm/s")
        return vel_vol, hdr

    @staticmethod
    def _parse_frame_order(ds, nframes: int):
        """
        Determine (Nz, Nt) from PerFrameFunctionalGroupsSequence or heuristic.

        Returns (nz, nt) where nz × nt == nframes.
        """
        if not hasattr(ds, "PerFrameFunctionalGroupsSequence"):
            # Fallback heuristic for data without PFGS
            for nt_guess in [17, 20, 25, 30, 1]:
                if nframes % nt_guess == 0:
                    nz = nframes // nt_guess
                    return nz, nt_guess
            return nframes, 1

        pfgs    = ds.PerFrameFunctionalGroupsSequence
        t_ids   = set()
        s_ids   = set()
        for frame in pfgs:
            # Temporal position
            try:
                t_ids.add(int(
                    frame.CardiacTriggerSequence[0]
                        .NominalCardiacTriggerDelayTime))
            except (AttributeError, IndexError, TypeError):
                try:
                    t_ids.add(int(
                        frame.TemporalPositionSequence[0]
                            .TemporalPositionTimeOffset))
                except (AttributeError, IndexError, TypeError):
                    pass
            # Spatial position (InStackPositionNumber or ImagePositionPatient z)
            try:
                s_ids.add(int(
                    frame.FrameContentSequence[0].InStackPositionNumber))
            except (AttributeError, IndexError, TypeError):
                try:
                    pos = frame.PlanePositionSequence[0].ImagePositionPatient
                    s_ids.add(round(float(pos[2]), 2))
                except (AttributeError, IndexError, TypeError):
                    pass

        nt = max(len(t_ids), 1)
        nz = max(len(s_ids), 1)

        # Sanity check
        if nt * nz != nframes:
            # Try factoring
            for nt_try in sorted([17, 20, 25, 30, nt], reverse=True):
                if nframes % nt_try == 0:
                    nt = nt_try
                    nz = nframes // nt
                    break

        return nz, nt

    @staticmethod
    def _detect_frame_order(px, nz: int, nt: int):
        """
        Auto-detect the frame ordering of a classic multiframe 4D-flow file that
        lacks per-frame position metadata (MeVisLab / xpace exports).

        Two conventions are possible:
          'phase_major'  frame = phase*Nz + slice   (slice index varies fastest)
          'slice_major'  frame = slice*Nt + phase   (phase index varies fastest)

        Picking the wrong one scrambles the slice positions, so Λ is applied at
        the wrong distance from isocentre and the per-slice correction shows a
        sawtooth. Frames of the SAME slice at adjacent cardiac phases are nearly
        identical (only the flow changes), so the correct layout is the one whose
        same-slice / adjacent-phase frames are most correlated. Each frame is
        block-averaged first to suppress the per-phase background noise that
        would otherwise mask the anatomy (important for phase images).

        Returns (order, margin) where margin is the correlation difference
        (bigger = more confident). Falls back to 'phase_major' when it cannot
        decide, since that is the ordering used by the MeVisLab/xpace exports
        this loader targets.
        """
        try:
            px = np.asarray(px, dtype=np.float32)
        except Exception:
            return "phase_major", 0.0
        if px.ndim != 3 or px.shape[0] != nz * nt or nz < 2 or nt < 2:
            return "phase_major", 0.0
        H, W = px.shape[1:]
        fy = max(1, H // 20); fx = max(1, W // 16)
        hy, hx = H // fy, W // fx
        sm = px[:, :hy * fy, :hx * fx].reshape(-1, hy, fy, hx, fx).mean(axis=(2, 4))

        def _corr(a, b):
            a = a.ravel() - a.mean(); b = b.ravel() - b.mean()
            d = np.sqrt(float((a * a).sum()) * float((b * b).sum()))
            return float((a * b).sum() / d) if d > 0 else 0.0

        # phase_major → reshape (Nt, Nz): same slice = adjacent along axis 0
        A = sm.reshape(nt, nz, hy, hx)
        cA = float(np.mean([_corr(A[p, s], A[p + 1, s])
                            for p in range(nt - 1) for s in range(nz)]))
        # slice_major → reshape (Nz, Nt): same slice = adjacent along axis 1
        B = sm.reshape(nz, nt, hy, hx)
        cB = float(np.mean([_corr(B[s, p], B[s, p + 1])
                            for s in range(nz) for p in range(nt - 1)]))
        order = "phase_major" if cA >= cB else "slice_major"
        return order, abs(cA - cB)

    @staticmethod
    def _build_frame_lut(ds, nframes: int, nz: int, nt: int,
                         order: str = "phase_major"):
        """
        Build a list of (slice_idx, time_idx) tuples, one per frame.

        Uses PerFrameFunctionalGroupsSequence when available; falls back
        to the simple row-major (frame = s*nt + t) ordering used by
        Siemens xpace exports.
        """
        lut = []

        if not hasattr(ds, "PerFrameFunctionalGroupsSequence"):
            # No per-frame geometry: use the ordering resolved by the caller
            # (auto-detected from the magnitude, or a manual override).
            if order == "slice_major":
                # frame = slice*Nt + phase  (phase index varies fastest)
                for fi in range(nframes):
                    lut.append((fi // nt, fi % nt))
            else:
                # phase_major (default): frame = phase*Nz + slice
                # (slice index varies fastest — the MeVisLab/xpace convention)
                for fi in range(nframes):
                    lut.append((fi % nz, fi // nz))
            return lut

        pfgs = ds.PerFrameFunctionalGroupsSequence

        # Collect unique temporal and spatial keys to build index maps
        t_keys = {}
        s_keys = {}

        for fi, frame in enumerate(pfgs):
            # Temporal key
            t_val = None
            try:
                t_val = float(
                    frame.CardiacTriggerSequence[0]
                        .NominalCardiacTriggerDelayTime)
            except (AttributeError, IndexError, TypeError):
                try:
                    t_val = float(
                        frame.TemporalPositionSequence[0]
                            .TemporalPositionTimeOffset)
                except (AttributeError, IndexError, TypeError):
                    t_val = fi % nt   # last-resort

            # Spatial key
            s_val = None
            try:
                s_val = int(
                    frame.FrameContentSequence[0].InStackPositionNumber) - 1
            except (AttributeError, IndexError, TypeError):
                try:
                    pos   = frame.PlanePositionSequence[0].ImagePositionPatient
                    s_val = round(float(pos[2]), 2)
                except (AttributeError, IndexError, TypeError):
                    s_val = fi // nt  # last-resort

            if t_val not in t_keys:
                t_keys[t_val] = len(t_keys)
            if s_val not in s_keys:
                s_keys[s_val] = len(s_keys)

            lut.append((s_keys[s_val], t_keys[t_val]))

        return lut

    @staticmethod
    def load_magnitude(directory: str) -> Optional[np.ndarray]:
        """
        Load a MAGNITUDE series (one image per slice) into a 3-D volume
        (Nx, Ny, Nz), used only to build a signal/lumen mask for the DICOM
        correction stats. Returns None on failure.
        """
        if not PYDICOM_AVAILABLE:
            return None
        try:
            # ── Single multiframe magnitude FILE (e.g. ww25_Magnitude_…dcm) ──
            # The Circle-of-Willis layout stores the magnitude as ONE multiframe
            # file (Nz×Nt frames) alongside the phase files, not a per-slice
            # series. Detect a file path and time-average its frames per slice.
            if os.path.isfile(directory):
                ds = pydicom.dcmread(directory)
                px = ds.pixel_array.astype(np.float32)      # (Nframes,Ny,Nx) or (Ny,Nx)
                if px.ndim == 2:
                    return px.T[:, :, None]
                nframes = int(px.shape[0])
                nz, nt  = DicomLoader._parse_frame_order(ds, nframes)
                ny, nx  = int(px.shape[1]), int(px.shape[2])
                # Resolve frame ordering (same logic/override as the phase loader)
                if DICOM_MULTIFRAME_ORDER in ("phase_major", "slice_major"):
                    _order = DICOM_MULTIFRAME_ORDER
                elif not hasattr(ds, "PerFrameFunctionalGroupsSequence"):
                    _order, _ = DicomLoader._detect_frame_order(px, nz, nt)
                else:
                    _order = "phase_major"
                acc = np.zeros((nx, ny, nz), np.float32)
                cnt = np.zeros(nz, np.float32)
                for fidx, (s_idx, t_idx) in enumerate(
                        DicomLoader._build_frame_lut(ds, nframes, nz, nt,
                                                     order=_order)):
                    if 0 <= s_idx < nz:
                        acc[:, :, s_idx] += px[fidx].T
                        cnt[s_idx] += 1.0
                cnt[cnt == 0] = 1.0
                mag = acc / cnt[None, None, :]               # time-averaged
                print(f"  [DICOM] Magnitude file loaded ({nx}×{ny}×{nz}, "
                      f"time-averaged over {nt} frame(s))")
                return mag

            files = DicomLoader.find_dicoms(directory)
            if not files:
                return None
            recs = []
            for f in files:
                h = pydicom.dcmread(f, stop_before_pixels=True)
                loc = getattr(h, "SliceLocation", None)
                if loc is None and hasattr(h, "ImagePositionPatient"):
                    loc = h.ImagePositionPatient[2]
                recs.append((round(float(loc if loc is not None else 0.0), 2),
                             int(getattr(h, "InstanceNumber", 0)), f))
            locs = sorted(set(r[0] for r in recs))
            ds0 = pydicom.dcmread(files[0], stop_before_pixels=True)
            nx, ny, nz = int(ds0.Columns), int(ds0.Rows), len(locs)
            mag = np.zeros((nx, ny, nz), dtype=np.float32)
            by_loc = {}
            for loc, inst, f in recs:
                by_loc.setdefault(loc, (inst, f))
            for s_i, loc in enumerate(locs):
                _, f = by_loc[loc]
                mag[:, :, s_i] = pydicom.dcmread(f).pixel_array.astype(np.float32).T
            print(f"  [DICOM] Magnitude mask series loaded ({nx}×{ny}×{nz})")
            return mag
        except Exception as e:
            print(f"  [DICOM] Could not load magnitude series: {e}")
            return None

    @staticmethod
    def _detect_direction_axes(files_by_frame, iop):
        """
        Work out which physical axis each of the three encoding frames measures.

        Returns (axes, description) where axes[d] is the axis index
        (0=x, 1=y, 2=z in DICOM patient coordinates) that frame d encodes,
        or (None, reason) if the headers do not say.

        Siemens writes the encoding direction into SequenceName as a suffix
        after the VENC value, e.g. 'fl3d1_v150rl' / '_v150ap' / '_v150in'.
        Anatomical suffixes map straight onto the LPS patient axes; the logical
        ones (readout / phase / slice) are resolved via ImageOrientationPatient.
        """

        # LPS convention: +x = Left, +y = Posterior, +z = Superior (Head)
        ANATOMIC = {"rl": 0, "lr": 0,        # right-left        -> x
                    "ap": 1, "pa": 1,        # anterior-posterior -> y
                    "fh": 2, "hf": 2}        # foot-head          -> z

        row_dir = col_dir = sl_dir = None
        if iop is not None and len(iop) == 6:
            row_dir = np.asarray(iop[0:3], float)
            col_dir = np.asarray(iop[3:6], float)
            sl_dir = np.cross(row_dir, col_dir)

        def _axis_of(vec):
            # Snaps the direction to its dominant axis. Exact for
            # axis-aligned stacks; for oblique stacks the discarded
            # transverse part mixes components. Alignment is checked and a
            # warning printed below when it drops under cos(5°) — the
            # general fix is the full rotation v_grad = R·v_patient, not a
            # nearest-axis assignment.
            v = np.asarray(vec, float)
            v = v / max(np.linalg.norm(v), 1e-12)
            ax_i = int(np.argmax(np.abs(v)))
            align = float(np.abs(v[ax_i]))
            if align < np.cos(np.deg2rad(5.0)):
                print(f"  [DICOM] WARNING: encoding direction {np.round(v,3)}"
                      f" snapped to axis '{'xyz'[ax_i]}' with only "
                      f"{np.degrees(np.arccos(min(1.0, align))):.1f}° "
                      "misalignment tolerance exceeded — oblique encoding; "
                      "component mixing is NOT corrected by this mapping.")
            return ax_i

        axes, labels = [], []
        for d, f in enumerate(files_by_frame):
            if f is None:
                return None, "no file found for one of the three frames"
            try:
                h = pydicom.dcmread(f, stop_before_pixels=True, force=True)
            except Exception:
                return None, "headers unreadable"

            seq = str(getattr(h, "SequenceName", "")).lower()
            m = re.search(r"v\d+([a-z]{1,2})$", seq)
            if not m:
                return None, "SequenceName carries no direction suffix"
            suf = m.group(1)

            if suf in ANATOMIC:
                axes.append(ANATOMIC[suf]); labels.append(suf)
            elif suf in ("in", "tp", "s") and sl_dir is not None:
                axes.append(_axis_of(sl_dir)); labels.append(suf)
            elif suf == "r" and row_dir is not None:
                axes.append(_axis_of(row_dir)); labels.append(suf)
            elif suf == "p" and col_dir is not None:
                axes.append(_axis_of(col_dir)); labels.append(suf)
            else:
                return None, f"unrecognised direction suffix '{suf}'"

        if sorted(axes) != [0, 1, 2]:
            return None, (f"suffixes {labels} do not map to three distinct "
                          f"axes (got {axes})")
        return axes, "  ".join(f"frame{d}:'{labels[d]}'->v{'xyz'[axes[d]]}"
                               for d in range(3))

    @staticmethod
    def _try_load_interleaved(files: List[str]):
        """
        Load 3-direction phase-contrast 4D-flow data where the three
        velocity-encoding directions are stored as separate frames in ONE
        series. Handles two layouts:

          • CARDIAC (pulsatile): directions interleaved in TriggerTime. A
            series with N_cardiac phases appears as 3·N_cardiac frames whose
            sorted unique TriggerTimes form N_cardiac tight triplets
            (e.g. [37.5,45,52.5] [120,127.5,135] …). → (3,Nx,Ny,Nz,N_cardiac)

          • STEADY (non-pulsatile): one time point, TriggerTime = 0, exactly 3
            files per slice, the directions distinguished by InstanceNumber
            blocks (1..Ns, Ns+1..2Ns, 2Ns+1..3Ns). → (3,Nx,Ny,Nz,1)

        Frame order 0/1/2 → vx/vy/vz (acquisition order — permute downstream if
        the physical axis assignment differs). Phase→velocity goes through the
        SAME phase_to_velocity() used by the multiframe path (one call per
        direction on the assembled raw volume): the phase full-scale is
        resolved from the rescale attributes — for typical Siemens exports
        (RS = 2, RI = −4096) that is exactly the fixed 4096 the earlier
        hardcoded conversion assumed, so results are unchanged there, while
        other calibrations (unipolar storage, non-4096 full-scale) are now
        handled instead of silently mis-scaled. RescaleSlope/Intercept must
        be constant within each direction; a mismatch raises rather than
        mixing two calibrations in one volume.

        Returns (vel_vol, hdr) or None if neither layout is detected (caller
        then falls back to _load_singleframe).
        """
        if not PYDICOM_AVAILABLE:
            return None
        ds0 = pydicom.dcmread(files[0], stop_before_pixels=True)

        # Geometry: (SliceLocation, TriggerTime, InstanceNumber) per file.
        # When SliceLocation is absent, the fallback is the IPP PROJECTED
        # ONTO THE SLICE NORMAL, s_k = r_k · n̂ — not IPP[2], which only
        # orders axial stacks (an oblique stack can share one z while the
        # slices differ along n̂).
        _n_hat0 = None
        if hasattr(ds0, "ImageOrientationPatient"):
            _io0 = [float(v) for v in ds0.ImageOrientationPatient]
            if len(_io0) == 6:
                _n_hat0 = np.cross(np.asarray(_io0[0:3], float),
                                   np.asarray(_io0[3:6], float))
        recs = []
        for f in files:
            h = pydicom.dcmread(f, stop_before_pixels=True)
            loc = getattr(h, "SliceLocation", None)
            if loc is None and hasattr(h, "ImagePositionPatient"):
                _p = np.asarray([float(v) for v in h.ImagePositionPatient],
                                float)
                loc = (float(np.dot(_p, _n_hat0)) if _n_hat0 is not None
                       else _p[2])
            recs.append((round(float(loc if loc is not None else 0.0), 2),
                         float(getattr(h, "TriggerTime", 0.0)),
                         int(getattr(h, "InstanceNumber", 0)), f))

        locs  = sorted(set(r[0] for r in recs))
        trigs = sorted(set(r[1] for r in recs))
        n_sl, n_tr = len(locs), len(trigs)
        if n_sl == 0 or len(files) % n_sl != 0:
            return None
        G = len(files) // n_sl                      # frames per slice
        if G % 3 != 0:
            return None

        # ── Decide layout ────────────────────────────────────────────────
        mode = None
        if n_tr >= 6 and n_tr % 3 == 0:
            tr   = np.asarray(trigs, dtype=float)
            gaps = np.diff(tr)
            n_grp = n_tr // 3
            big = set(np.argsort(gaps)[-(n_grp - 1):]) if n_grp > 1 else set()
            if n_grp == 1 or set(range(2, n_tr - 1, 3)).issubset(big):
                mode, Nt = "cardiac", n_grp
                groups = tr.reshape(Nt, 3)
        if mode is None and n_tr <= 1 and G == 3:
            mode, Nt = "steady", 1
        if mode is None:
            return None

        venc   = DicomLoader._extract_venc(files)
        nx, ny = int(ds0.Columns), int(ds0.Rows)
        vel    = np.zeros((3, nx, ny, n_sl, Nt), dtype=np.float32)

        # Collect RAW stored pixels per direction; conversion happens ONCE
        # per direction through the shared phase_to_velocity() so every
        # loader resolves the phase full-scale and bipolar/unipolar zero
        # point identically (see docstring).
        raw    = np.zeros((3, nx, ny, n_sl, Nt), dtype=np.float64)
        _cal   = [set(), set(), set()]     # (slope, intercept) per direction
        _bits  = [None, None, None]

        def _raw(f, d):
            ds    = pydicom.dcmread(f)
            px    = ds.pixel_array.astype(np.float64)
            _cal[d].add((float(getattr(ds, "RescaleSlope",     1.0)),
                         float(getattr(ds, "RescaleIntercept", 0.0))))
            if _bits[d] is None:
                _bits[d] = getattr(ds, "BitsStored", None)
            return px.T

        if mode == "cardiac":
            idx = {(l, round(t, 2)): f for l, t, _, f in recs}
            for s_i, loc in enumerate(locs):
                for p_i in range(Nt):
                    for d in range(3):
                        f = idx.get((loc, round(groups[p_i, d], 2)))
                        if f is not None:
                            raw[d, :, :, s_i, p_i] = _raw(f, d)
            detail = f"{Nt} cardiac phases × 3 directions = {n_tr} frames"
        else:  # steady — 3 dirs per slice, ordered by InstanceNumber
            by_slice = defaultdict(list)
            for loc, tt, inst, f in recs:
                by_slice[loc].append((tt, inst, f))
            for s_i, loc in enumerate(locs):
                grp = sorted(by_slice[loc])           # by (trigger, instance)
                for d in range(3):
                    raw[d, :, :, s_i, 0] = _raw(grp[d][2], d)
            detail = f"steady (1 time point) × 3 directions = {G} frames/slice"

        # ── Shared phase→velocity conversion, one call per direction ─────
        for d in range(3):
            if len(_cal[d]) != 1:
                raise ValueError(
                    f"Interleaved frame {d}: RescaleSlope/Intercept differ "
                    f"across its files ({sorted(_cal[d])}). One direction "
                    "must carry ONE phase calibration — mixing two would "
                    "scale slices inconsistently. Check the series for "
                    "mixed reconstructions.")
            _sl, _in = next(iter(_cal[d]))
            vel[d] = DicomLoader.phase_to_velocity(
                raw[d], _sl, _in, venc, label=f"frame{d}",
                bits_stored=_bits[d]).astype(np.float32)
        del raw

        px_sp = [float(v) for v in getattr(ds0, "PixelSpacing", [1.0, 1.0])]
        sl_th = float(getattr(ds0, "SliceThickness", 1.0))
        _iop = ([float(v) for v in ds0.ImageOrientationPatient]
                if hasattr(ds0, "ImageOrientationPatient") else None)

        print(f"  [DICOM] Format: 3-direction phase — {mode} ({detail})")
        print(f"  [DICOM] De-interleaved → (3×{nx}×{ny}×{n_sl}×{Nt})  "
              f"VENC={venc*100:.0f} cm/s")

        # ── Assign the three frames to physical velocity axes ─────────────
        # The frame order alone says nothing about which axis each frame
        # encodes. Λ is anisotropic (Λxx, Λyy, Λzz all differ), so a permuted
        # component would be corrected with the wrong matrix element. Resolve
        # it from the headers where possible, and say plainly which route was
        # taken so the assumption is never silent.
        _axes = None
        _dir_src = "assumed"          # 'manual' | 'header' | 'assumed'
        if DICOM_DIRECTION_ORDER is not None:
            _spec = str(DICOM_DIRECTION_ORDER).strip().lower()
            if sorted(_spec) != ["x", "y", "z"]:
                raise ValueError(
                    "DICOM_DIRECTION_ORDER must be a permutation of 'xyz' "
                    f"(e.g. 'xyz', 'zxy'), got {DICOM_DIRECTION_ORDER!r}")
            _axes = ["xyz".index(c) for c in _spec]
            _dir_src = "manual"
            print(f"  [DICOM] Direction order: '{_spec}' set manually "
                  "(DICOM_DIRECTION_ORDER) — "
                  + ", ".join(f"frame{d}→v{_spec[d]}" for d in range(3)))
        else:
            # First frame of the first cardiac phase / first slice, per direction
            if mode == "cardiac":
                _probe = [idx.get((locs[0], round(groups[0, d], 2)))
                          for d in range(3)]
            else:
                _g = sorted(by_slice[locs[0]])
                _probe = [_g[d][2] if d < len(_g) else None for d in range(3)]
            _axes, _why = DicomLoader._detect_direction_axes(_probe, _iop)
            if _axes is not None:
                _dir_src = "header"
                print("  [DICOM] Direction order detected from SequenceName: "
                      f"{_why}")
            else:
                print("  [DICOM] Direction order: could NOT be detected "
                      f"({_why}).")
                print("           Falling back to acquisition order "
                      "frame 0→vx, 1→vy, 2→vz.")
                print("           VERIFY THIS. In a phantom the flow direction "
                      "is known: if the fluid runs")
                print("           through-plane, vz should carry nearly all the "
                      "speed inside the lumen.")
                print("           Set DICOM_DIRECTION_ORDER (e.g. 'zxy') if the "
                      "components are permuted.")

        if _axes is not None and _axes != [0, 1, 2]:
            vel = vel[np.argsort(_axes)]      # frame d holds axis _axes[d]
            print("  [DICOM] Velocity components permuted to (vx, vy, vz).")

        # ── Origin of voxel (0,0,0), and slice-direction sign ─────────────
        # The caller reconstructs positions as
        #     origin + i*row_dir*sp + j*col_dir*sp + k*cross(row,col)*thk
        # so the origin must be the FIRST slice in the array, and the array's
        # slice axis must run along +cross(row_dir, col_dir). Slices were
        # ordered by ascending SliceLocation, which is not guaranteed to match
        # that sign — if it does not, the volume would be built mirrored in z.
        _ipp0 = None
        try:
            _ipp_by_loc = {}
            for _l, _t, _i, _f in recs:
                if _l not in _ipp_by_loc:
                    _h = pydicom.dcmread(_f, stop_before_pixels=True, force=True)
                    if hasattr(_h, "ImagePositionPatient"):
                        _ipp_by_loc[_l] = [float(v) for v in
                                           _h.ImagePositionPatient]
            if len(_ipp_by_loc) >= 2 and _iop is not None:
                _first = np.asarray(_ipp_by_loc[locs[0]], float)
                _last = np.asarray(_ipp_by_loc[locs[-1]], float)
                _norm = np.cross(np.asarray(_iop[0:3], float),
                                 np.asarray(_iop[3:6], float))
                if float(np.dot(_last - _first, _norm)) < 0:
                    vel = vel[:, :, :, ::-1, :]        # flip the slice axis
                    _ipp0 = list(_last)
                    print("  [DICOM] Slice order flipped to match the slice "
                          "normal (volume was mirrored in z).")
                else:
                    _ipp0 = list(_first)
            elif _ipp_by_loc:
                _ipp0 = list(_ipp_by_loc[locs[0]])
        except Exception as _exc:
            print(f"  [DICOM] Could not resolve the volume origin ({_exc}); "
                  "the caller will centre it on isocentre.")
            _ipp0 = None

        if _ipp0 is not None:
            _r_off = float(np.linalg.norm(np.asarray(_ipp0, float)))
            print("  [DICOM] Volume origin from header: "
                  f"({_ipp0[0]:.1f}, {_ipp0[1]:.1f}, {_ipp0[2]:.1f}) mm "
                  f"({_r_off:.0f} mm from isocentre)")

        # ── True slice-to-slice step ──────────────────────────────────────
        # SliceThickness is the EXCITATION width; the geometric step between
        # slice centres can differ (gapped or overlapping slices). Prefer the
        # IPP span projected onto the slice normal, then the spread of
        # SliceLocation values; the caller falls back to SliceThickness when
        # neither is available.
        _sl_step_mm = None
        try:
            if (_ipp0 is not None and _iop is not None
                    and len(_ipp_by_loc) >= 2):
                _n_hat = np.cross(np.asarray(_iop[0:3], float),
                                  np.asarray(_iop[3:6], float))
                _p_a = np.asarray(_ipp_by_loc[locs[0]], float)
                _p_b = np.asarray(_ipp_by_loc[locs[-1]], float)
                _sl_step_mm = (abs(float(np.dot(_p_b - _p_a, _n_hat)))
                               / max(len(_ipp_by_loc) - 1, 1))
            elif len(locs) >= 2:
                _sl_step_mm = float(np.median(np.abs(np.diff(
                    np.asarray(sorted(set(locs)), float)))))
        except Exception:
            _sl_step_mm = None

        hdr = {
            "VENC":              venc,
            "PixelSpacing_mm":   px_sp,
            "SliceThickness_mm": sl_th,
            "SliceSpacing_mm":   _sl_step_mm,
            "Nt":                Nt,
            "Nslices":           n_sl,
            "format":            "interleaved3dir",
            # which patient axis each ACQUIRED frame was assigned (frame 0,1,2)
            "direction_axes":    ("".join("xyz"[int(a)] for a in _axes)
                                  if _axes is not None else "xyz"),
            "direction_source":  _dir_src,
            # Geometry, so the caller places the volume at its TRUE position
            # instead of assuming isocentre. Gradient nonlinearity is zero at
            # isocentre and grows outwards, so an assumed position evaluates Λ
            # in the wrong place entirely.
            "ImagePositionPatient":      _ipp0,
            "ImageOrientationPatient":   _iop,
        }

        # ── Layout validity self-check ────────────────────────────────────
        # The de-interleaving above assumes a particular file ordering. If
        # that ordering is wrong (e.g. the files are BLOCKED — all of dir1,
        # then all of dir2, then all of dir3 — but were read as INTERLEAVED,
        # or vice-versa), then each output voxel combines the three components
        # of DIFFERENT physical voxels. The magnitude |v|=sqrt(vx²+vy²+vz²)
        # then routinely exceeds VENC, which is physically impossible for real
        # phase data. Measure that here and warn loudly, because no amount of
        # gradient correction can rescue a mis-assembled volume.
        try:
            _mag = np.sqrt((vel ** 2).sum(axis=0))       # (nx,ny,nz,Nt)
            # Restrict to voxels that actually carry SIGNAL. Background air is
            # uniform random phase, so each component is ~uniform on
            # [-VENC,+VENC] and |v| exceeds VENC for 1 - pi/6 = 47.6% of it —
            # a pure geometry fact (the fraction of a cube outside its
            # inscribed sphere), independent of any layout error. Measuring
            # over the WHOLE volume therefore reports 25-45% on any phantom
            # with air around it and this warning fires every time, which is
            # worse than useless: it cries wolf on correct data.
            # No magnitude volume is available here, but temporal coherence
            # separates the two: real flow keeps a consistent direction across
            # cardiac phases, noise does not (its time-average shrinks by
            # sqrt(Nt)).
            _finite = _mag[np.isfinite(_mag)]
            if _finite.size:
                _frac_over = float((_finite > venc).mean())
                hdr["layout_frac_over_venc_unmasked"] = _frac_over
                # Informational ONLY. Do not warn from this number: with air
                # included it is ~0.476 x (air fraction) even for perfectly
                # assembled data, i.e. 25-45% on any phantom. The real check
                # needs the magnitude mask and runs later, in main().
                print(f"  [DICOM] Phase assembly: {100*_frac_over:.0f}% of ALL "
                      f"voxels exceed VENC (air alone gives ~48% x air "
                      f"fraction;\n          not diagnostic on its own — the "
                      f"masked check follows once magnitude is loaded).")
        except Exception as _e:
            print(f"  [DICOM] Layout check skipped: {_e}")

        return vel, hdr

    @staticmethod
    def _load_singleframe(files: List[str]) -> Tuple[np.ndarray, dict]:
        """
        Classic single-frame 4D-flow export: one 2-D DICOM file per
        (encoding direction, slice[, cardiac phase]), the three directions
        stored as SEPARATE SERIES — or as one series whose files carry
        three distinct SequenceName velocity-encoding suffixes.

        Until this method existed, load_series() advertised this fallback
        but calling it raised AttributeError — the hard bug flagged by the
        external code review. The layout it covers is exactly the one
        _try_load_interleaved cannot detect: with three per-direction
        series sharing the same TriggerTimes, the unique trigger count is
        Nt (not 3·Nt), so neither the cardiac-triplet nor the
        3-files-per-slice steady pattern matches and the interleaved
        detector correctly returns None.

        Reconstruction:
          1. group files by SeriesInstanceUID (fallback: by SequenceName
             velocity suffix within a single series);
          2. map each group to its physical axis with the same
             _detect_direction_axes used by the interleaved loader
             (DICOM_DIRECTION_ORDER overrides, as everywhere else);
          3. within each direction sort slices by IPP·n̂ — the projection
             onto the slice normal, valid for oblique stacks where IPP[2]
             ties — and cardiac phases by TriggerTime;
          4. convert phase→velocity ONCE per direction through the shared
             phase_to_velocity() (constant RescaleSlope/Intercept per
             direction enforced; a mismatch raises rather than mixing two
             calibrations in one volume).

        Returns (vel, hdr) with the same hdr keys as the other loaders.
        Raises ValueError with a folder inventory — never AttributeError —
        when three directions cannot be identified.
        """
        if not PYDICOM_AVAILABLE:
            raise RuntimeError("pydicom is required for DICOM loading.")

        hdrs = []
        for f in files:
            try:
                hdrs.append((f, pydicom.dcmread(f, stop_before_pixels=True,
                                                force=True)))
            except Exception as e:
                raise ValueError(f"Unreadable DICOM file {f!r}: {e}")

        def _suffix(h):
            m = re.search(r"v\d+([a-z]{1,2})$",
                          str(getattr(h, "SequenceName", "")).lower())
            return m.group(1) if m else None

        # ── 1. Group the files into candidate directions ─────────────────
        by_series = defaultdict(list)
        for f, h in hdrs:
            by_series[str(getattr(h, "SeriesInstanceUID", "?"))].append((f, h))

        groups, grouping = None, ""
        if len(by_series) == 3:
            groups = list(by_series.values())
            grouping = "SeriesInstanceUID (3 series)"
        else:
            by_suffix = defaultdict(list)
            for f, h in hdrs:
                by_suffix[_suffix(h)].append((f, h))
            by_suffix.pop(None, None)
            if len(by_suffix) == 3:
                groups = list(by_suffix.values())
                grouping = "SequenceName suffix (one series, 3 suffixes)"
        if groups is None:
            inv = "; ".join(
                f"…{uid[-8:]}: {len(v)} files, "
                f"SequenceName {str(getattr(v[0][1], 'SequenceName', '?'))!r}"
                for uid, v in by_series.items())
            raise ValueError(
                "Classic single-frame layout needs THREE encoding "
                "directions — either three series, or three SequenceName "
                f"velocity suffixes within one series. Found "
                f"{len(by_series)} series [{inv}]. A 4D-flow PHASE folder "
                "holds three times the files of its magnitude twin; check "
                "that the phase series (all three directions) was "
                "selected, or use the interleaved/multiframe layouts "
                "handled automatically.")

        ds0  = hdrs[0][1]
        _iop = ([float(v) for v in ds0.ImageOrientationPatient]
                if hasattr(ds0, "ImageOrientationPatient") else None)
        n_hat = None
        if _iop is not None and len(_iop) == 6:
            n_hat = np.cross(np.asarray(_iop[0:3], float),
                             np.asarray(_iop[3:6], float))

        def _loc(h):
            # Slice coordinate: IPP PROJECTED ONTO THE SLICE NORMAL
            # (s_k = r_k · n̂) — the oblique-safe ordering; IPP[2] only
            # orders axial stacks.
            if hasattr(h, "ImagePositionPatient") and n_hat is not None:
                p = np.asarray([float(v) for v in h.ImagePositionPatient],
                               float)
                return round(float(np.dot(p, n_hat)), 2)
            sl = getattr(h, "SliceLocation", None)
            if sl is not None:
                return round(float(sl), 2)
            if hasattr(h, "ImagePositionPatient"):
                return round(float(h.ImagePositionPatient[2]), 2)
            return 0.0

        # ── 2. Consistent (slice, phase) lattice across the groups ──────
        locs  = sorted({_loc(h) for g in groups for _, h in g})
        trigs = sorted({float(getattr(h, "TriggerTime", 0.0))
                        for g in groups for _, h in g})
        n_sl = len(locs)
        Nt   = max(len(trigs), 1)
        for gi, g in enumerate(groups):
            if len(g) != n_sl * Nt:
                raise ValueError(
                    f"Direction group {gi} holds {len(g)} files, but the "
                    f"folder implies {n_sl} slices × {Nt} cardiac phases "
                    f"= {n_sl * Nt} per direction. Incomplete or mixed "
                    "series?")

        nx, ny = int(ds0.Columns), int(ds0.Rows)
        raw   = np.zeros((3, nx, ny, n_sl, Nt), dtype=np.float64)
        _cal  = [set(), set(), set()]
        _bits = [None, None, None]
        s_ix  = {l: i for i, l in enumerate(locs)}
        t_ix  = {t: i for i, t in enumerate(trigs)}

        for gi, g in enumerate(groups):
            for f, h in g:
                ds = pydicom.dcmread(f)
                _cal[gi].add((float(getattr(ds, "RescaleSlope", 1.0)),
                              float(getattr(ds, "RescaleIntercept", 0.0))))
                if _bits[gi] is None:
                    _bits[gi] = getattr(ds, "BitsStored", None)
                _ti = t_ix.get(float(getattr(h, "TriggerTime", 0.0)), 0)
                raw[gi, :, :, s_ix[_loc(h)], _ti] = \
                    ds.pixel_array.astype(np.float64).T

        venc = DicomLoader._extract_venc(files)
        vel  = np.zeros((3, nx, ny, n_sl, Nt), dtype=np.float32)
        # Shared phase→velocity conversion, one call per direction — same
        # resolver as the multiframe and interleaved paths.
        for d in range(3):
            if len(_cal[d]) != 1:
                raise ValueError(
                    f"Direction group {d}: RescaleSlope/Intercept differ "
                    f"across its files ({sorted(_cal[d])}). One direction "
                    "must carry ONE phase calibration — mixing two would "
                    "scale slices inconsistently.")
            _sl, _in = next(iter(_cal[d]))
            vel[d] = DicomLoader.phase_to_velocity(
                raw[d], _sl, _in, venc, label=f"dir{d}",
                bits_stored=_bits[d]).astype(np.float32)
        del raw

        print(f"  [DICOM] Format: classic single-frame — grouped by "
              f"{grouping}")
        print(f"  [DICOM] Assembled → (3×{nx}×{ny}×{n_sl}×{Nt})  "
              f"VENC={venc*100:.0f} cm/s")

        # ── 3. Frame → physical-axis assignment (same policy as the ─────
        #      interleaved loader: manual override, else header suffixes,
        #      else keep order with a loud VERIFY message)
        _axes = None
        if DICOM_DIRECTION_ORDER is not None:
            _spec = str(DICOM_DIRECTION_ORDER).strip().lower()
            if sorted(_spec) != ["x", "y", "z"]:
                raise ValueError(
                    "DICOM_DIRECTION_ORDER must be a permutation of "
                    f"'xyz' (e.g. 'xyz', 'zxy'), got "
                    f"{DICOM_DIRECTION_ORDER!r}")
            _axes = ["xyz".index(c) for c in _spec]
            print(f"  [DICOM] Direction order: '{_spec}' set manually "
                  "(DICOM_DIRECTION_ORDER) — "
                  + ", ".join(f"group{d}→v{_spec[d]}" for d in range(3)))
        else:
            _probe = [g[0][0] for g in groups]
            _axes, _why = DicomLoader._detect_direction_axes(_probe, _iop)
            if _axes is not None:
                print("  [DICOM] Direction order detected from "
                      f"SequenceName: {_why}")
            else:
                print(f"  [DICOM] Direction order could NOT be detected "
                      f"({_why}).")
                print("           Keeping group order as vx, vy, vz — "
                      "VERIFY THIS, or set DICOM_DIRECTION_ORDER.")
        if _axes is not None and _axes != [0, 1, 2]:
            vel = vel[np.argsort(_axes)]      # group d holds axis _axes[d]
            print("  [DICOM] Velocity components permuted to (vx, vy, vz).")

        # ── 4. Geometry for the caller ───────────────────────────────────
        # locs ascend along +n̂ BY CONSTRUCTION (projection sort), so the
        # first slice is the volume origin and, unlike the interleaved
        # loader, no mirror check is needed.
        g0_by_loc = {}
        for f, h in groups[0]:
            g0_by_loc.setdefault(_loc(h), h)
        h_first = g0_by_loc[locs[0]]
        _ipp0 = ([float(v) for v in h_first.ImagePositionPatient]
                 if hasattr(h_first, "ImagePositionPatient") else None)
        if _ipp0 is not None:
            _r_off = float(np.linalg.norm(np.asarray(_ipp0, float)))
            print("  [DICOM] Volume origin from header: "
                  f"({_ipp0[0]:.1f}, {_ipp0[1]:.1f}, {_ipp0[2]:.1f}) mm "
                  f"({_r_off:.0f} mm from isocentre)")
        _sl_step = (float(locs[-1] - locs[0]) / (n_sl - 1)
                    if n_sl > 1 else None)
        px_sp = [float(v) for v in getattr(ds0, "PixelSpacing", [1.0, 1.0])]
        sl_th = float(getattr(ds0, "SliceThickness", 1.0))

        hdr = {
            "VENC":              venc,
            "PixelSpacing_mm":   px_sp,
            "SliceThickness_mm": sl_th,
            "SliceSpacing_mm":   _sl_step,
            "Nt":                Nt,
            "Nslices":           n_sl,
            "format":            "singleframe_classic",
            "ImagePositionPatient":      _ipp0,
            "ImageOrientationPatient":   _iop,
        }
        return vel, hdr


# ═════════════════════════════════════════════════════════════════════════════
# PHASE ERROR CORRECTIONS
# ═════════════════════════════════════════════════════════════════════════════


def _verify_signs_anatomically(v5, Xmm, Ymm, Zmm, inside, mag=None):
    """
    Check the loaded velocity SIGNS against intracranial anatomy.

    WHY THIS EXISTS. The sign of a phase-contrast velocity is a vendor
    convention, carried by Velocity Encoding Direction (0018,9090) or the
    MR Velocity Encoding Sequence (0018,9197). Research exports often strip
    both, leaving DICOM_DIRECTION_SIGNS to be set by hand. An inverted sign
    is then INVISIBLE in every result this toolbox reports, because |v|,
    |dv| = |v_cor - v_meas| and angle(v_meas, v_cor) are all invariant under
    v -> -v. Only the flow DIRECTION reveals it, so the direction is what
    this function tests.

    Six textbook facts about the circle of Willis, in DICOM LPS
    (+x left, +y posterior, +z superior):
        left  MCA  flows laterally  -> vx > 0
        right MCA  flows laterally  -> vx < 0
        ICA terminus ascends        -> vz > 0
        basilar ascends             -> vz > 0
        ACA runs anteriorly         -> vy < 0
        PCA runs posteriorly        -> vy > 0

    Anatomy is a PRIOR, not a measurement, so nothing is changed here: the
    agreement fraction per axis is reported and a replacement sign tuple is
    suggested when a component is consistently inverted.

    Returns a dict of the per-axis agreement fractions, or None.
    """
    v = v5.mean(axis=-1) if v5.ndim == 5 else v5          # (3,Nx,Ny,Nz)
    spd = np.sqrt((v ** 2).sum(axis=0))
    # Restrict to CONFIDENT vessel voxels. A loose mask dilutes the anatomical
    # signal with parenchyma, whose direction is noise, and the agreement
    # fractions drift towards 50% for every axis. Two independent vessel
    # criteria are combined, the same pair the figures use:
    #   - bright in the magnitude image (inflow enhancement), taken at the
    #     99th percentile so the cut adapts to the dataset's intensity scale
    #   - temporally coherent, i.e. the direction holds across the cycle
    sel = inside & (spd > 0.15)                            # > 15 cm/s
    _vessel = None
    if v5.ndim == 5 and v5.shape[-1] >= 4:
        _sp_all = np.sqrt((v5 ** 2).sum(axis=0))
        _coh = (np.sqrt((v5.mean(axis=-1) ** 2).sum(axis=0))
                / np.maximum(_sp_all.mean(axis=-1), 1e-12))
        _vessel = (_coh >= 0.80)
    if mag is not None and np.asarray(mag).shape == spd.shape:
        _m = np.asarray(mag, float)
        _sig = _m > 0.10 * float(_m.max())
        if _sig.sum() > 1000:
            _bright = _m >= float(np.percentile(_m[_sig], 99.0))
            # OR, not AND: the two criteria are independent and each finds
            # vessels the other misses (bright-but-slow vs coherent-but-dim).
            _vessel = _bright if _vessel is None else (_vessel | _bright)
    if _vessel is not None:
        sel = sel & _vessel
    if int(sel.sum()) < 200:
        print("  [DICOM] anatomical sign check: too few fast vessel voxels "
              f"({int(sel.sum())}) — skipped.")
        return None

    aX = np.abs(Xmm)
    tests = [
        ("left  MCA  vx>0", sel & (Xmm > 22) & (Ymm > -60) & (Ymm < -35), 0, +1),
        ("right MCA  vx<0", sel & (Xmm < -22) & (Ymm > -60) & (Ymm < -35), 0, -1),
        ("ICA term.  vz>0", sel & (aX > 7) & (aX < 22) & (Ymm > -52) & (Ymm < -32), 2, +1),
        ("basilar    vz>0", sel & (aX < 8) & (Ymm > -32) & (Ymm < -12), 2, +1),
        ("ACA        vy<0", sel & (aX < 10) & (Ymm < -52), 1, -1),
        ("PCA        vy>0", sel & (aX > 12) & (aX < 32) & (Ymm > -30) & (Ymm < -12), 1, +1),
    ]
    print("  [DICOM] ── Anatomical sign check (intracranial) ──")
    agree = {0: [], 1: [], 2: []}
    for name, m, comp, exp in tests:
        n = int(m.sum())
        if n < 50:
            print(f"           {name}: only {n} voxels — skipped")
            continue
        c = v[comp][m]
        f = float((c > 0).mean() if exp > 0 else (c < 0).mean())
        agree[comp].append(f)
        print(f"           {name}   n={n:>6,}   agreement {100*f:5.1f} %"
              f"   {'ok' if f > 0.65 else 'INVERTED' if f < 0.35 else 'unclear'}")

    cur = list(DICOM_DIRECTION_SIGNS) if DICOM_DIRECTION_SIGNS else [1, 1, 1]
    flip, out = [], {}
    for a in range(3):
        if not agree[a]:
            flip.append(False); out["xyz"[a]] = None; continue
        f = float(np.mean(agree[a]))
        out["xyz"[a]] = f
        flip.append(f < 0.35)
    if any(flip):
        rec = tuple(int(-cur[a]) if flip[a] else int(cur[a]) for a in range(3))
        print("  " + "!" * 68)
        print("  [DICOM] The velocity SIGNS disagree with anatomy on "
              + ", ".join("v" + "xyz"[a] for a in range(3) if flip[a]) + ".")
        print(f"           Current DICOM_DIRECTION_SIGNS = "
              f"{tuple(int(c) for c in cur)}")
        print(f"           Anatomy is consistent with     = {rec}")
        print("           A wrong sign does NOT change |v|, |dv| or the "
              "correction angle, so it is")
        print("           invisible in the summary figures — but every "
              "statement about flow")
        print("           DIRECTION, and every glyph or streamline, is "
              "reversed. Set the tuple")
        print("           above and re-run before quoting directional "
              "results.")
        print("  " + "!" * 68)
    else:
        print("           Signs are consistent with anatomy on every "
              "testable axis.")
    out["recommended_signs"] = (
        [int(-cur[a]) if flip[a] else int(cur[a]) for a in range(3)]
        if any(flip) else [int(c) for c in cur])
    return out


def write_velocity_vtk(path: str,
                       v_meas: np.ndarray,
                       v_corr: np.ndarray,
                       spacing_mm,
                       origin_mm=(0.0, 0.0, 0.0),
                       mag=None,
                       venc_cm_s=None,
                       t_index=None,
                       lean=None,
                       coherence=None,
                       speed_timeavg_cm_s=None):
    """
    Write measured and corrected 4D-flow velocity fields to a legacy VTK
    STRUCTURED_POINTS file that ParaView can open directly.
    The file stores, per voxel:
      - velocity_measured   (3-component vector, cm/s)
      - velocity_corrected  (3-component vector, cm/s)
      - correction          (corrected − measured, 3-vector, cm/s)
      - |v|_measured, |v|_corrected, |correction|  (scalars, cm/s)
      - magnitude           (the anatomical magnitude, if provided)
    In ParaView you can then, e.g., add a Glyph filter for velocity vectors,
    or a Stream Tracer for streamlines through the vessel/pipe.
    Parameters
    ----------
    path : str
        Output .vtk path. When t_index="all", this is used as a STEM: the
        files written are <stem>_t00.vtk, <stem>_t01.vtk, … which ParaView
        loads as a single animatable time series.
    v_meas, v_corr : np.ndarray, shape (3, Nx, Ny, Nz, Nt) or (3, Nx, Ny, Nz)
        Velocity fields in m/s. See t_index for how the time axis is handled.
    spacing_mm : sequence of 3 floats
        Voxel size (dx, dy, dz) in mm.
    origin_mm : sequence of 3 floats
        Physical position of voxel (0,0,0), in mm.
    mag : np.ndarray, shape (Nx, Ny, Nz), optional
        Anatomical magnitude for masking/context in ParaView.
    venc_cm_s : float, optional
        Written into the file header comment for reference.
    t_index : int | "all" | None
        Which cardiac phase to write. None (default) auto-selects PEAK
        SYSTOLE — the phase with the highest mean speed inside the object.
        This used to be hardcoded to t=0, which on gated data is usually a
        near-diastolic phase: the export then looked almost empty in
        ParaView and the streamlines would not seed, for no visible reason.
        Pass an int to force a phase, or "all" for the full time series.
    lean : bool, optional
        Write only the two velocity vectors, omitting the arrays ParaView
        can derive itself (speeds = the "Magnitude" component of a vector;
        correction = a Calculator on the two vectors) and the static anatomy
        (written once to <stem>_anatomy.vtk instead of repeated per phase).
        Default: True for "all", False for a single phase.
    Returns
    -------
    str for a single phase, or list[str] for t_index="all" (the per-phase
    files, followed by the anatomy file when one is written).
    Notes
    -----
    Uses only numpy — no external VTK dependency — so it runs anywhere the
    main program runs. Data is written big-endian binary, as the legacy VTK
    binary format requires.
    """
    vm_full = np.asarray(v_meas, dtype=np.float32)
    vc_full = np.asarray(v_corr, dtype=np.float32)

    # ── Resolve the time axis ────────────────────────────────────────────
    if vm_full.ndim == 5:
        nt = vm_full.shape[-1]
        if isinstance(t_index, str) and t_index.lower() == "all":
            lean_ts = True if lean is None else bool(lean)
            stem    = str(Path(path).with_suffix(""))
            npts    = int(np.prod(vm_full.shape[1:4]))
            per_f   = npts * (24 if lean_ts else 52)     # bytes per phase
            need    = per_f * nt + (npts * 4 if mag is not None else 0)

            def _sz(n):
                return (f"{n/1024**3:.1f} GB" if n >= 1024**3
                        else f"{n/1024**2:.0f} MB")

            # Pre-flight disk check. Writing gigabytes into a full drive
            # leaves a truncated file that ParaView opens as garbage, and on
            # Windows it can fill the system volume — fail before that, not
            # after 14 of 17 phases.
            # The probe and the raise are deliberately SEPARATE: IOError IS
            # OSError in Python 3, so raising inside the try would be caught
            # by its own handler and silently discarded.
            try:
                free = shutil.disk_usage(str(Path(path).parent)).free
            except (OSError, ValueError):
                free = None                # unknown filesystem — proceed
            if free is not None and free < need * 1.10:
                raise IOError(
                    f"VTK time series needs ~{_sz(need)} but only "
                    f"{_sz(free)} is free on the output drive. Free up space, "
                    "choose an output folder on a bigger drive, or set "
                    "DICOM_VTK_PHASES = \"peak\".")

            print(f"  [vtk] Time series: {nt} phases x "
                  f"{_sz(per_f)} = ~{_sz(need)} "
                  f"({'lean: 2 velocity vectors' if lean_ts else 'full array set'} "
                  "per phase)")
            paths = []
            for _t in range(nt):
                paths.append(write_velocity_vtk(
                    f"{stem}_t{_t:02d}.vtk", v_meas, v_corr, spacing_mm,
                    origin_mm=origin_mm, mag=None, venc_cm_s=venc_cm_s,
                    t_index=_t, lean=lean_ts))
                print(f"  [vtk]   phase {_t+1:2d}/{nt} written", end="\r",
                      flush=True)

            # Static anatomy, written ONCE. In ParaView: open it as a second
            # source alongside the series and use it for context (Volume
            # rendering or a Contour), or as the Glyph/Stream Tracer mask.
            if mag is not None:
                m = np.asarray(mag, dtype=np.float32)
                if m.shape == tuple(vm_full.shape[1:4]):
                    a_path = f"{stem}_anatomy.vtk"
                    _write_scalar_vtk(a_path, m, spacing_mm, origin_mm,
                                      name="anatomy_magnitude")
                    paths.append(a_path)

            print(f"  [vtk] Wrote {nt} phases as a ParaView time series: "
                  f"{Path(stem).name}_t00.vtk … _t{nt-1:02d}.vtk"
                  + ("  (+ _anatomy.vtk)" if mag is not None else ""))
            return paths
        if t_index is None:
            if nt == 1:
                # Steady flow exported as a single phase. There is no cardiac
                # cycle to find a peak in, so say so plainly rather than
                # announcing "peak systole" and then warning that the speed
                # does not vary — with one phase it cannot.
                t_use = 0
                print("  [vtk] Single phase (Nt=1) — steady flow, no cardiac "
                      "peak to select.")
            else:
                # Peak systole. This must be measured on the VESSELS, not on the
                # bulk of the masked volume. In vivo the signal mask is ~99%
                # static tissue whose speed is noise, so a plain mean over it is
                # nearly flat with respect to cardiac phase (a ~0.7 cm/s bump on
                # a ~6.6 cm/s baseline) and its argmax is set by whatever tissue
                # signal happens to vary — that is how this landed on t=0, the
                # phase that carries the peak flow.
                # So: find the fastest voxels by TIME-AVERAGED speed (that is the
                # vessel tree), then ask which phase those voxels peak in. Their
                # systolic swing is tens of cm/s, so the argmax is unambiguous.
                spd  = np.linalg.norm(vm_full, axis=0)         # (Nx,Ny,Nz,Nt)
                tavg = spd.mean(axis=-1)
                sel  = None
                if mag is not None and np.asarray(mag).shape == spd.shape[:3]:
                    m = np.asarray(mag, dtype=np.float32)
                    sel = m > 0.10 * float(m.max())
                    if not sel.any():
                        sel = None
                pool = tavg[sel] if sel is not None else tavg
                ves  = ((sel if sel is not None else np.ones_like(tavg, bool))
                        & (tavg >= np.percentile(pool, 99.0)))
                if ves.sum() < 32:                  # degenerate mask — fall back
                    ves = tavg >= np.percentile(tavg, 99.9)
                curve = spd[ves].mean(axis=0) * 100.0     # m/s -> cm/s for display
                t_use = int(np.argmax(curve))
                print(f"  [vtk] Writing cardiac phase t={t_use} of {nt} "
                      f"(peak systole, from {int(ves.sum()):,} vessel voxels: "
                      f"{curve.min():.1f}–{curve.max():.1f} cm/s over the cycle).")
                if curve.max() - curve.min() < 0.05 * max(curve.max(), 1e-6):
                    print("  [vtk] WARNING: vessel speed barely varies across "
                          "phases — the data may not be cardiac-gated, or the "
                          "phase axis may be misordered. Check before quoting "
                          "this as peak systole.")
        else:
            t_use = int(t_index) % nt
        vm = vm_full[..., t_use]
        vc = vc_full[..., t_use] if vc_full.ndim == 5 else vc_full
    else:
        vm, vc = vm_full, vc_full
    lean = bool(lean)                     # None -> False for a single phase
    # vm, vc now (3, Nx, Ny, Nz)
    _, nx, ny, nz = vm.shape
    dvec = (vc - vm)                        # correction vector
    # Convert m/s → cm/s for readable ParaView units.
    vm_cm = vm * 100.0
    vc_cm = vc * 100.0
    dv_cm = dvec * 100.0
    sx, sy, sz = [float(s) for s in spacing_mm]
    ox, oy, oz = [float(o) for o in origin_mm]
    # VTK STRUCTURED_POINTS orders point data with X fastest, then Y, then Z.
    # Our arrays are (component, x, y, z); we need, per point in (x,y,z) with
    # x fastest, the components. Build (Npts, 3) with the right ordering.
    def _vec_block(arr_cm):
        # arr_cm: (3, nx, ny, nz) → (nx*ny*nz, 3) with x fastest (Fortran order
        # over spatial axes), components last.
        a = np.transpose(arr_cm, (1, 2, 3, 0))          # (nx,ny,nz,3)
        a = np.transpose(a, (2, 1, 0, 3))               # (nz,ny,nx,3) so x fastest on ravel
        return a.reshape(-1, 3).astype(">f4")           # big-endian for VTK
    def _scalar_block(arr_cm):
        a = np.transpose(arr_cm, (2, 1, 0))             # (nz,ny,nx)
        return a.reshape(-1).astype(">f4")
    mag_vm = np.linalg.norm(vm_cm, axis=0)
    mag_vc = np.linalg.norm(vc_cm, axis=0)
    mag_dv = np.linalg.norm(dv_cm, axis=0)
    npts = nx * ny * nz
    with open(path, "wb") as f:
        hdr = []
        hdr.append("# vtk DataFile Version 3.0")
        _c = "4D flow velocity (cm/s)"
        if venc_cm_s:
            _c += f", VENC={venc_cm_s:.0f} cm/s"
        # The field is UNMASKED: air voxels carry uniform random phase and
        # average ~VENC in |v|. Say so in the one header line every VTK
        # reader shows, so nobody quotes statistics from the raw file.
        _c += " - unmasked; threshold on anatomy_magnitude for stats"
        hdr.append(_c[:255])
        hdr.append("BINARY")
        hdr.append("DATASET STRUCTURED_POINTS")
        hdr.append(f"DIMENSIONS {nx} {ny} {nz}")
        hdr.append(f"ORIGIN {ox:.6g} {oy:.6g} {oz:.6g}")
        hdr.append(f"SPACING {sx:.6g} {sy:.6g} {sz:.6g}")
        hdr.append(f"POINT_DATA {npts}")
        f.write(("\n".join(hdr) + "\n").encode("ascii"))
        def _write_vectors(name, block):
            f.write(f"VECTORS {name} float\n".encode("ascii"))
            f.write(block.tobytes())
            f.write(b"\n")
        def _write_scalars(name, block):
            f.write(f"SCALARS {name} float 1\nLOOKUP_TABLE default\n"
                    .encode("ascii"))
            f.write(block.tobytes())
            f.write(b"\n")
        _write_vectors("velocity_measured",  _vec_block(vm_cm))
        _write_vectors("velocity_corrected", _vec_block(vc_cm))
        if not lean:
            # Derivable in ParaView (Calculator / the vector "Magnitude"
            # colour component), so a time series omits them — see
            # DICOM_VTK_LEAN. A single-phase export keeps them for
            # convenience, since the size does not matter there.
            _write_vectors("correction",      _vec_block(dv_cm))
            _write_scalars("speed_measured",  _scalar_block(mag_vm))
            _write_scalars("speed_corrected", _scalar_block(mag_vc))
            _write_scalars("correction_mag",  _scalar_block(mag_dv))
            # ── Arrays that let ParaView separate FLOW from NOISE ─────────
            # A single exported phase cannot do this: |v| of stationary
            # tissue is rectified noise averaging sqrt(8/pi)*sigma, so a
            # speed threshold low enough to keep slow vessels also keeps
            # parenchyma. These two scalars are computed over ALL cardiac
            # phases and are therefore available even though one phase is
            # exported:
            #   flow_coherence = |mean_t(v)| / mean_t(|v|)
            #       ~1        for real flow (direction held across the cycle)
            #       ~1/sqrt(Nt) for noise (direction random each phase)
            #   speed_timeavg  = mean_t(|v|), sqrt(Nt) less noisy than the
            #       single-phase speed already written above.
            # Threshold on flow_coherence in ParaView to keep SLOW vessels
            # while rejecting tissue — which no speed threshold can do.
            if coherence is not None:
                _write_scalars("flow_coherence",
                               _scalar_block(np.asarray(coherence,
                                                        dtype=np.float64)))
            if speed_timeavg_cm_s is not None:
                _write_scalars("speed_timeavg",
                               _scalar_block(np.asarray(speed_timeavg_cm_s,
                                                        dtype=np.float64)))
            if mag is not None:
                m = np.asarray(mag, dtype=np.float32)
                if m.shape == (nx, ny, nz):
                    _write_scalars("anatomy_magnitude", _scalar_block(m))
    return path


def _write_scalar_vtk(path: str, vol: np.ndarray, spacing_mm,
                      origin_mm=(0.0, 0.0, 0.0), name: str = "scalar") -> str:
    """
    Write a single static 3-D scalar volume as legacy VTK STRUCTURED_POINTS.

    Used for the anatomy magnitude that accompanies a velocity TIME SERIES.
    The magnitude volume is time-averaged upstream, so it is identical in
    every cardiac phase; writing it once instead of Nt times saves ~290 MB
    on a 17-phase 260x320x52 export and loses nothing.
    """
    v = np.asarray(vol, dtype=np.float32)
    nx, ny, nz = v.shape
    sx, sy, sz = [float(s) for s in spacing_mm]
    ox, oy, oz = [float(o) for o in origin_mm]
    with open(path, "wb") as f:
        f.write(("\n".join([
            "# vtk DataFile Version 3.0",
            "static anatomy magnitude",
            "BINARY",
            "DATASET STRUCTURED_POINTS",
            f"DIMENSIONS {nx} {ny} {nz}",
            f"ORIGIN {ox:.6g} {oy:.6g} {oz:.6g}",
            f"SPACING {sx:.6g} {sy:.6g} {sz:.6g}",
            f"POINT_DATA {nx*ny*nz}",
        ]) + "\n").encode("ascii"))
        f.write(f"SCALARS {name} float 1\nLOOKUP_TABLE default\n"
                .encode("ascii"))
        f.write(np.transpose(v, (2, 1, 0)).reshape(-1).astype(">f4").tobytes())
        f.write(b"\n")
    return path


# ═════════════════════════════════════════════════════════════════════════════
# GRADIENT NONLINEARITY VELOCITY CORRECTOR
# ═════════════════════════════════════════════════════════════════════════════

class VelocityCorrector:
    """
    Applies the generalized phase-contrast velocity reconstruction of
    Markl et al. (2003), Eq. [7], voxel-wise to a 4D velocity field:

        v_true(r, t) = [Λ(r)ᵀ]⁻¹ · v_measured(r, t)

    Recovering the full velocity VECTOR (magnitude and direction) requires
    three-directional encoding, as Markl et al. note; with single-direction
    data only the magnitude term of their Eq. [11]/[12] can be corrected.

    Uses batched LAPACK matrix inversion (numpy.linalg.inv) with
    Einstein summation for vectorised application across all voxels.
    Singular voxels (det < threshold) are left uncorrected.
    """

    def __init__(self, cfg: SimulationConfig):
        self.cfg = cfg

    def correct_4d(
        self,
        v_measured: np.ndarray,
        Lambda: np.ndarray,
    ) -> np.ndarray:
        """
        Voxel-wise gradient nonlinearity correction across all time frames.

        Parameters
        ----------
        v_measured : (3, Nx, Ny, Nz, Nt)
        Lambda : (3, 3, Nx, Ny, Nz)

        Returns
        -------
        v_corrected : same shape as v_measured
        """
        shape        = v_measured.shape[1:-1]   # (Nx, Ny, Nz)
        Nt           = v_measured.shape[-1]
        spatial_size = int(np.prod(shape))

        # Reshape for batch inversion: (N_vox, 3, 3)
        L_flat   = Lambda.reshape(3, 3, spatial_size).transpose(2, 0, 1)

        # ── Invertibility gate on the smallest singular value ────────────
        # det(Λ) = σ1·σ2·σ3 can stay far above any threshold while σ_min → 0
        # (e.g. σ = (100, 1, 1e-6) gives det = 1e-4), so a determinant
        # threshold does not measure conditioning. 1/σ_min IS the worst-case
        # noise amplification of the inversion, so the gate is placed on it
        # directly. One batched SVD supplies σ_min, σ_max and κ = σ_max/σ_min
        # — the adaptive-Tikhonov branch below reuses κ instead of running a
        # second SVD through np.linalg.cond. (cfg.det_threshold is retained
        # in the config for backward compatibility but no longer used.)
        _sv      = np.linalg.svd(L_flat, compute_uv=False)   # (N_vox, 3) desc
        sig_max  = _sv[:, 0]
        sig_min  = _sv[:, -1]
        _thr     = float(getattr(self.cfg, "sigma_min_threshold", 1e-4))
        valid    = sig_min > _thr
        n_sing   = (~valid).sum()
        if n_sing > 0:
            print(f"  [corrector] {n_sing} voxels masked as non-invertible "
                  f"(σ_min ≤ {_thr:g}; {100*n_sing/spatial_size:.2f}%)")

        # Batch inversion of valid voxels
        L_inv_flat        = np.zeros_like(L_flat)

        lam = float(getattr(self.cfg, "tikhonov_lambda", 0.0))
        use_adaptive = bool(getattr(self.cfg, "tikhonov_adaptive", True))

        if lam > 0.0:
            I3 = np.eye(3)
            Lv = L_flat[valid]                                  # (N_valid, 3, 3)

            if use_adaptive:
                # ── ADAPTIVE Tikhonov regularised pseudoinverse ─────────
                # Per-voxel λ scales with the local condition number κ(Λ):
                # κ ≤ κ_low → λ_local = 0 (exact inverse, NO bias)
                # κ ≥ κ_high → λ_local = λ_max (full regularisation)
                # between → linear ramp
                #
                # NAMING — this is condition-number-adaptive Tikhonov
                # regularisation, a HEURISTIC introduced here as an
                # extension of the Markl reconstruction. It is NOT an
                # L-curve method: the L-curve selects λ from the corner of
                # the residual-norm vs solution-norm trade-off curve traced
                # as λ varies; a direct κ→λ ramp constructs no such curve.
                # Tikhonov regularisation is the standard remedy for the
                # ill-posed inversion that Markl et al. (2003) handle with
                # SVD (their Appendix, Eqs. [A6]–[A7]); report this ramp as
                # a custom heuristic.
                # In practice, with κ(Λ) ≈ 1 everywhere inside R0 on
                # near-isocentre data, κ ≤ κ_low holds for essentially every
                # voxel and the scheme reduces to the exact Markl inverse
                # (the printout below reports the exact-inverse fraction).
                kappa_low  = float(getattr(self.cfg, "tikhonov_kappa_low",  2.0))
                kappa_high = float(getattr(self.cfg, "tikhonov_kappa_high", 10.0))
                # κ from the singular values already computed by the gate
                # above (σ_max/σ_min) — identical to np.linalg.cond(Lv) for
                # the 2-norm, without a second SVD pass.
                kappa = (sig_max / np.maximum(sig_min, 1e-300))[valid]
                ramp = (kappa - kappa_low) / max(kappa_high - kappa_low, 1e-9)
                lam_local = lam * np.clip(ramp, 0.0, 1.0)
                lam_sq    = (lam_local * lam_local)[:, None, None]
                LtL = np.einsum("...ji,...jk->...ik", Lv, Lv) + lam_sq * I3
                Lt  = np.einsum("...ji->...ij", Lv)
                L_inv_flat[valid] = np.linalg.solve(LtL, Lt)
                n_exact = int((lam_local == 0).sum())
                n_full  = int((lam_local >= lam * 0.99).sum())
                print("  [corrector] Adaptive Tikhonov (threshold)  "
                      f"λ_max={lam:.3f}  κ_low={kappa_low}  κ_high={kappa_high}")
                print(f"              exact inverse: {n_exact} voxels  "
                      f"({100*n_exact/len(Lv):.1f}%)  |  "
                      f"full λ: {n_full} voxels  "
                      f"({100*n_full/len(Lv):.1f}%)")
            else:
                # ── Fixed-λ Tikhonov regularised pseudoinverse ──────────
                # Λ_reg⁻¹ = (Λᵀ·Λ + λ²·I)⁻¹ · Λᵀ
                # Related to the truncated-SVD inverse — both are spectral
                # filters — but NOT mathematically equivalent: Tikhonov's
                # filter factors are σ²/(σ²+λ²) (a smooth roll-off around
                # σ ≈ λ), while TSVD applies a hard 0/1 cutoff. They give
                # different inverses for any finite λ.
                LtL = np.einsum("...ji,...jk->...ik", Lv, Lv) + (lam * lam) * I3
                Lt  = np.einsum("...ji->...ij", Lv)
                L_inv_flat[valid] = np.linalg.solve(LtL, Lt)
                print(f"  [corrector] Fixed-λ Tikhonov inverse  λ = {lam:.3f}")
        else:
            # Original exact inverse (numerically unstable at FOV edges)
            L_inv_flat[valid] = np_inv(L_flat[valid])        # LAPACK batch
        # Fallback: identity for non-invertible voxels. The ARRAY stays
        # finite (visualisation, VTK export), but these voxels carry the
        # UNCORRECTED velocity and are NOT valid corrections. The mask is
        # exposed as self.last_valid_mask so quantitative consumers can
        # exclude them explicitly; the pipeline's inside-R0 mask already
        # does so in practice — on near-isocentre data Λ ≈ I keeps
        # σ_min ≈ 1 everywhere inside the calibration sphere, so the
        # fallback fires only in the extrapolated shell that every metric
        # masks anyway.
        L_inv_flat[~valid] = np.eye(3)
        self.last_valid_mask = valid.reshape(shape)

        # Reshape back: (3, 3, Nx, Ny, Nz)
        L_inv = L_inv_flat.transpose(1, 2, 0).reshape((3, 3) + shape)

        # Apply correction for each time frame.
        # Forward model: v_meas = Λᵀ · v_true (apply_forward_distortion)
        # Correction: v_cor = (Λᵀ)⁻¹ · v_meas = (Λ⁻¹)ᵀ · v_meas
        # einsum "ji..." performs the transpose-of-inverse contraction:
        # v_cor_i = Σ_j (L_inv)_ji · v_meas_j
        # so that v_cor = (Λ⁻¹)ᵀ · Λᵀ · v_true = v_true ✓
        # This convention exactly matches the 3D reference simulation
        # (generalized_recon → Lambda_inv_T) and Markl 2003
        v_cor = np.zeros_like(v_measured)
        for t in range(Nt):
            v_cor[..., t] = np.einsum(
                "ji...,j...->i...", L_inv, v_measured[..., t]
            )

        return v_cor


# ═════════════════════════════════════════════════════════════════════════════
# VALIDATION METRICS ENGINE
# ═════════════════════════════════════════════════════════════════════════════

class ValidationEngine:
    """
    Computes all quantitative validation metrics:

        · RMSE, MAE (velocity magnitude, [m/s])
        · Pearson r (component-wise and magnitude)
        · Flow rate Q (volumetric, [ml/min])
        · Radial stratification in 10 mm bins
        · Percentage improvement factor
        · Bland–Altman bias and limits of agreement
    """

    @staticmethod
    def rmse(
        v_est: np.ndarray,
        v_ref: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> float:
        """
        Root Mean Square Error of velocity magnitude.

        When `mask` is provided , the metric is restricted to
        the masked voxels — typically the inside_r0_mask, so that voxels
        outside the spherical-harmonic calibration sphere are excluded.

        Vector-error convention
        ----------------------------------
        RMSE is now the root-mean-square of the velocity-ERROR VECTOR
        magnitude ||v_est − v_ref||, NOT the magnitude difference
        |‖v_est‖ − ‖v_ref‖|. The two differ whenever the error has a
        component perpendicular to the flow (e.g. helical swirl): the
        magnitude-difference form silently hides such cross-component
        errors. This makes rmse(), mae() and the bootstrap CI consistent
        with radial_stratification() (fig08), which already used the
        vector-error form. (Bland–Altman intentionally keeps the
        magnitude-difference form — it is a scalar agreement analysis.)
        """
        err_sq = np.sum((v_est - v_ref) ** 2, axis=0)   # ||v_est − v_ref||²
        if mask is not None and mask.any():
            err_sq = err_sq[mask]
        return float(np.sqrt(np.mean(err_sq)))

    @staticmethod
    def mae(
        v_est: np.ndarray,
        v_ref: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> float:
        """Mean Absolute Error of the velocity error vector, optionally masked.

        Uses ||v_est − v_ref|| (vector error), consistent with
        rmse(). See rmse() docstring for the rationale.
        """
        err_mag = np.linalg.norm(v_est - v_ref, axis=0)
        if mask is not None and mask.any():
            err_mag = err_mag[mask]
        return float(np.mean(err_mag))

    @staticmethod
    def pearson(
        v_est: np.ndarray,
        v_ref: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> Tuple[float, float, float, float]:
        """
        Pearson correlation for each component + magnitude, optionally
        restricted to voxels inside `mask`.
        Returns (r_x, r_y, r_z, r_mag).

        Report the PER-COMPONENT values, not r_mag. Pearson's r is invariant
        under linear scaling, and gradient nonlinearity is a multiplicative
        distortion (v_meas = Lambda(r) . v_true). Where Lambda is roughly
        constant across the ROI, r_mag is ~1 both before and after correction
        by construction and carries no information. The per-component values
        respond to the off-diagonal Lambda terms that mix x/y/z, which is the
        directional error that magnitude cancels out.

        Guard against zero-variance arrays (all-zero reference outside ROI)
        which cause pearsonr to return nan.
        """
        rs = []
        # Per-component correlation
        for ci in range(3):
            a = v_est[ci]
            b = v_ref[ci]
            if mask is not None:
                a = a[mask]; b = b[mask]
            a = a.ravel(); b = b.ravel()
            nz = (np.abs(a) > 1e-12) | (np.abs(b) > 1e-12)
            if nz.sum() < 10 or np.std(b[nz]) < 1e-12:
                # Too few voxels or no variance: the correlation is undefined.
                # Return NaN, not 1.0 — a degenerate input must never be
                # reported as a perfect correlation.
                rs.append(float("nan"))
            else:
                r, _ = pearsonr(a[nz], b[nz])
                rs.append(float(r) if np.isfinite(r) else float("nan"))
        # Magnitude correlation
        mag_e = np.linalg.norm(v_est, axis=0)
        mag_r = np.linalg.norm(v_ref, axis=0)
        if mask is not None:
            mag_e = mag_e[mask]; mag_r = mag_r[mask]
        mag_e = mag_e.ravel(); mag_r = mag_r.ravel()
        nz_m  = (mag_r > 1e-12) | (mag_e > 1e-12)
        if nz_m.sum() < 10 or np.std(mag_r[nz_m]) < 1e-12:
            r_mag = float("nan")
        else:
            r_mag, _ = pearsonr(mag_e[nz_m], mag_r[nz_m])
            r_mag = float(r_mag) if np.isfinite(r_mag) else float("nan")
        rs.append(r_mag)
        return tuple(rs)

    @staticmethod
    def flow_rate_ml_min(
        v_field: np.ndarray,
        tube_mask: np.ndarray,
        voxel_area_m2: float,
    ) -> float:
        """
        Volumetric flow rate Q = ∫∫ v_x · dA over the tube cross-section.
        Averaged over time frames. Returns Q in ml/min.

        Algorithm
        ---------------------------------------
        Summing v_x · ΔA over the full 3D tube cylinder would over-count
        by a factor of N_x (the number of x-slices the tube spans), so the
        integration uses a single representative cross-section:
          1. Find x-slices where the 2D y-z lumen mask is non-empty
             (the tube's extent along x).
          2. Integrate v_x over the y-z cross-section at the CENTRAL
             x-slice of that extent.
          3. Average over Nt cardiac phases.
          4. Convert m³/s → ml/min.

        For steady Poiseuille flow this gives Q exactly; for pulsatile
        flow it gives the cycle-mean Q at the chosen slice. If you
        need the flow rate at a specific slice (e.g. through a known
        analysis plane in the in-vivo Circle-of-Willis dataset), bias
        the tube_mask so that only that slice is non-empty.
        """
        # x-axis extent: any x-slice with at least one tube voxel
        x_axis_present = tube_mask.any(axis=(1, 2))            # (Nx,) bool
        if not x_axis_present.any():
            return 0.0                                          # degenerate
        x_indices = np.where(x_axis_present)[0]
        ix_mid    = x_indices[len(x_indices) // 2]              # central slice
        cs_mask   = tube_mask[ix_mid, :, :]                     # (Ny, Nz)

        Nt   = v_field.shape[-1]
        q_t  = np.array([
            v_field[0, ix_mid, :, :, t][cs_mask].sum() * voxel_area_m2
            for t in range(Nt)
        ])
        return float(q_t.mean() * 1e6 * 60)   # m³/s → ml/min

    @staticmethod
    def pct_improvement(rmse_before: float, rmse_after: float) -> float:
        """Percentage RMSE reduction after correction."""
        return 100.0 * (1.0 - rmse_after / max(rmse_before, 1e-12))

    @staticmethod
    def radial_stratification(
        v_est: np.ndarray,
        v_ref: np.ndarray,
        X: np.ndarray,
        Y: np.ndarray,
        Z: np.ndarray,
        bin_width_mm: float = 10.0,
        max_r_mm:     float = 220.0,
        vessel_mask:  Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Stratify RMSE by radial distance from isocenter in 10 mm bins
        .

        Parameters
        ----------
        vessel_mask : (Nx, Ny, Nz) bool, optional
            If given, only voxels where the mask is True are included in
            the RMSE per-bin calculation. Restricting to vessel voxels
            (|v_ref| > threshold) makes the metric physiologically
            meaningful — it answers "how well does the correction recover
            real flow?" rather than "how well does it preserve noise in
            empty space?".

        Returns (r_centres_mm, rmse_per_bin, voxel_count_per_bin)
        """
        r_mm    = np.sqrt(X**2 + Y**2 + Z**2) * 1000.0
        # Average over time if 5D
        if v_est.ndim == 5:
            ve = v_est.mean(axis=-1)
            vr = v_ref.mean(axis=-1)
        else:
            ve, vr = v_est, v_ref

        mag_err = np.linalg.norm(ve - vr, axis=0)

        edges   = np.arange(0, max_r_mm + bin_width_mm, bin_width_mm)
        centres = 0.5 * (edges[:-1] + edges[1:])
        rmse_b  = np.zeros(len(centres))
        count_b = np.zeros(len(centres), dtype=int)

        for k, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
            mask = (r_mm >= lo) & (r_mm < hi)
            if vessel_mask is not None:
                mask = mask & vessel_mask
            count_b[k]  = mask.sum()
            if count_b[k] > 0:
                rmse_b[k] = np.sqrt(np.mean(mag_err[mask]**2))

        return centres, rmse_b, count_b

    @staticmethod
    def bland_altman(
        v_meas: np.ndarray,
        v_ref:  np.ndarray,
        mask:   Optional[np.ndarray] = None,
    ) -> Tuple[float, float, float, np.ndarray, np.ndarray]:
        """
        Bland–Altman analysis on velocity magnitude, optionally masked.
        Returns (bias, loa_lo, loa_hi, means_flat, diffs_flat).
        """
        mag_m = np.linalg.norm(v_meas, axis=0)
        mag_r = np.linalg.norm(v_ref,  axis=0)
        if mask is not None:
            mag_m = mag_m[mask]
            mag_r = mag_r[mask]
        mag_m = mag_m.ravel()
        mag_r = mag_r.ravel()
        diff  = mag_m - mag_r
        mean  = 0.5 * (mag_m + mag_r)
        bias  = diff.mean()
        sd    = diff.std()
        return bias, bias - 1.96*sd, bias + 1.96*sd, mean, diff

    @staticmethod
    def rmse_bootstrap_ci(
        v_est:        np.ndarray,
        v_ref:        np.ndarray,
        n_bootstrap:  int   = 1000,
        ci_level:     float = 0.95,
        rng_seed:     int   = 42,
        mask:         Optional[np.ndarray] = None,
    ) -> Tuple[float, float, float]:
        """
        RMSE point estimate and bootstrap confidence interval.

        When `mask` is provided , the bootstrap is restricted
        to voxels inside the spherical-harmonic calibration sphere so the
        reported confidence intervals reflect only the physically reliable
        region of the FOV.

        Method (Efron & Tibshirani 1993, Chap. 13)
        -------------------------------------------
        1. Compute voxel-wise squared vector error e_i = ||v_est_i − v_ref_i||²
           (restricted to mask voxels if provided).
        2. Resample n_bootstrap times with replacement.
        3. Compute RMSE = sqrt(mean(e)) on each resample → bootstrap dist.
        4. CI = (lower-α/2, upper-α/2) percentiles of that distribution.

        Uses the vector error ||v_est − v_ref|| (consistent with
        rmse()/mae()), not the magnitude difference.

        Returns
        -------
        (rmse_pt, ci_low, ci_high) all in m/s.
        These are DESCRIPTIVE intervals for each RMSE separately. Comparing
        before/after through CI non-overlap is a conservative sufficient
        condition only (the errors live on the same voxels and are
        positively correlated, so the unpaired comparison discards power);
        the significance test to quote is the paired improvement CI from
        paired_improvement_bootstrap_ci().
        """
        rng     = np.random.default_rng(rng_seed)
        err_sq  = np.sum((v_est - v_ref) ** 2, axis=0)   # ||v_est − v_ref||²
        if mask is not None:
            err_sq = err_sq[mask]
        err_sq   = err_sq.ravel()
        n        = err_sq.size
        if n == 0:
            return 0.0, 0.0, 0.0

        # Point estimate
        rmse_pt = float(np.sqrt(np.mean(err_sq)))

        # Bootstrap distribution (resample squared errors, then RMS)
        boot_rmses = np.empty(n_bootstrap)
        for k in range(n_bootstrap):
            idx           = rng.integers(0, n, n)
            boot_rmses[k] = np.sqrt(np.mean(err_sq[idx]))

        alpha   = 1.0 - ci_level
        ci_low  = float(np.percentile(boot_rmses, 100 * alpha / 2.0))
        ci_high = float(np.percentile(boot_rmses, 100 * (1.0 - alpha / 2.0)))

        return rmse_pt, ci_low, ci_high

    @staticmethod
    def paired_improvement_bootstrap_ci(
        v_before:     np.ndarray,
        v_after:      np.ndarray,
        v_ref:        np.ndarray,
        n_bootstrap:  int   = 1000,
        ci_level:     float = 0.95,
        rng_seed:     int   = 42,
        mask:         Optional[np.ndarray] = None,
        block_index:  Optional[np.ndarray] = None,
    ) -> Tuple[float, float, float, float, float]:
        """
        Paired bootstrap CI of the RMSE improvement (Efron & Tibshirani
        1993), optionally with spatial block resampling.

        WHY PAIRED. The before- and after-correction errors are computed on
        the SAME voxels with the same noise realisation, so they are
        strongly positively correlated. Bootstrapping them separately and
        comparing CI endpoints (the previous approach) is a valid but
        conservative sufficient condition that discards the pairing and
        with it most of the power; the interval it produced for the
        improvement (built from opposing CI endpoints) is a worst-case
        bound, not a CI. Here every bootstrap replicate draws ONE index set
        and evaluates the paired statistics inside the replicate:

            Δ_b = RMSE_after,b − RMSE_before,b            [m/s]
            I_b = 100 · (1 − RMSE_after,b / RMSE_before,b)  [%]

        and the percentile CIs of Δ_b and I_b are returned. Significance at
        the (1 − ci_level) level: the Δ CI excludes 0 (equivalently the
        improvement CI excludes 0).

        WHY BLOCKS. Neighbouring voxels share the smooth Λ field, partial-
        volume structure and reconstruction correlations, so voxel-wise
        i.i.d. resampling overstates the effective sample size and narrows
        every CI. Pass `block_index` — an integer label per masked voxel
        (e.g. the slice index along the vessel axis) — to resample whole
        blocks with replacement instead of voxels; this is the standard
        remedy for spatially correlated fields and also shrinks the index
        volume by roughly the block size.

        Returns
        -------
        (impr_pt_pct, impr_lo_pct, impr_hi_pct, delta_lo_ms, delta_hi_ms)
        """
        rng = np.random.default_rng(rng_seed)
        eb  = np.sum((v_before - v_ref) ** 2, axis=0)
        ea  = np.sum((v_after  - v_ref) ** 2, axis=0)
        if mask is not None:
            eb, ea = eb[mask], ea[mask]
        eb, ea = eb.ravel(), ea.ravel()
        n = eb.size
        if n == 0:
            return 0.0, 0.0, 0.0, 0.0, 0.0

        rmse_b_pt = float(np.sqrt(np.mean(eb)))
        rmse_a_pt = float(np.sqrt(np.mean(ea)))
        impr_pt   = 100.0 * (1.0 - rmse_a_pt / max(rmse_b_pt, 1e-300))

        boot_impr  = np.empty(n_bootstrap)
        boot_delta = np.empty(n_bootstrap)

        if block_index is not None:
            blk = np.asarray(block_index).ravel()
            if blk.size != n:
                raise ValueError(
                    f"block_index has {blk.size} entries for {n} masked "
                    "voxels — it must label exactly the masked voxels.")
            labels = np.unique(blk)
            # Pre-split the paired errors per block once; each replicate
            # then just concatenates resampled blocks.
            eb_blocks = [eb[blk == b] for b in labels]
            ea_blocks = [ea[blk == b] for b in labels]
            n_blk = len(labels)
            for k in range(n_bootstrap):
                pick = rng.integers(0, n_blk, n_blk)
                ebk = np.concatenate([eb_blocks[i] for i in pick])
                eak = np.concatenate([ea_blocks[i] for i in pick])
                rb  = np.sqrt(ebk.mean())
                ra  = np.sqrt(eak.mean())
                boot_delta[k] = ra - rb
                boot_impr[k]  = 100.0 * (1.0 - ra / max(rb, 1e-300))
        else:
            for k in range(n_bootstrap):
                idx = rng.integers(0, n, n)
                rb  = np.sqrt(eb[idx].mean())
                ra  = np.sqrt(ea[idx].mean())
                boot_delta[k] = ra - rb
                boot_impr[k]  = 100.0 * (1.0 - ra / max(rb, 1e-300))

        a = 1.0 - ci_level
        return (impr_pt,
                float(np.percentile(boot_impr,  100 * a / 2.0)),
                float(np.percentile(boot_impr,  100 * (1 - a / 2.0))),
                float(np.percentile(boot_delta, 100 * a / 2.0)),
                float(np.percentile(boot_delta, 100 * (1 - a / 2.0))))


# ═════════════════════════════════════════════════════════════════════════════
# PLOTTING ENGINE
# ═════════════════════════════════════════════════════════════════════════════

class PlottingEngine:
    """
    Generates the diagnostic and validation figures.
    Saves PNG files (300 dpi) to the specified output directory.
    """

    def __init__(self, output_dir: str, coil_name: str = ""):
        self.out   = Path(output_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.coil  = coil_name
    def _save(self, fig, name: str):
        # Honour the ENABLED_FIGS whitelist: parse the figure number from the
        # filename (e.g. "fig02_lambda_matrix.png" → 2) and skip saving figures
        # that are not selected. Selected figures are then renamed to their
        # sequential position via FIG_RENUMBER (e.g. fig17 → fig03).
        _num = None
        if name.startswith("fig"):
            _stem = name[3:].split("_")[0]
            if _stem.isdigit():
                _num = int(_stem)
        if ENABLED_FIGS is not None and _num is not None and _num not in ENABLED_FIGS:
            plt.close(fig)
            return
        if _num is not None and _num in FIG_RENUMBER:
            _suffix = name[3:].split("_", 1)[1] if "_" in name[3:] else name[3:]
            name = f"fig{FIG_RENUMBER[_num]:02d}_{_suffix}"
        p = self.out / name
        fig.savefig(p, bbox_inches="tight", dpi=150)
        plt.close(fig)
        print(f"  [fig] Saved → {p.name}")

    @staticmethod
    def _imshow_2d(ax, data, extent, title, cmap, vmin=None, vmax=None,
                   vcenter=None, unit="", fontsize=8):
        if vcenter is not None:
            # Guard: TwoSlopeNorm requires strict vmin < vcenter < vmax.
            # When the slice is all-zero the supplied limits collapse to 0,0,0
            # which raises ValueError. We add a small epsilon to guarantee
            # the strict ordering even for flat fields.
            # NaN-safe: masked (extrapolated) voxels are stored as NaN so the
            # colourmap can render them grey. A plain .max()/.min() would
            # return NaN and TwoSlopeNorm would then raise.
            _amax = float(np.nanmax(np.abs(data))) if np.any(np.isfinite(data)) else 0.0
            eps = max(_amax * 1e-3, 1e-9)
            _dmin = float(np.nanmin(data)) if np.any(np.isfinite(data)) else 0.0
            _dmax = float(np.nanmax(data)) if np.any(np.isfinite(data)) else 0.0
            _vmin    = min(float(vmin)    if vmin    is not None else _dmin, vcenter - eps)
            _vmax    = max(float(vmax)    if vmax    is not None else _dmax, vcenter + eps)
            _vcenter = float(vcenter)
            # Final safety clamp
            if _vmin >= _vcenter:
                _vmin = _vcenter - eps
            if _vmax <= _vcenter:
                _vmax = _vcenter + eps
            norm = TwoSlopeNorm(vmin=_vmin, vcenter=_vcenter, vmax=_vmax)
            im = ax.imshow(data, origin="lower", extent=extent,
                           cmap=cmap, norm=norm, aspect="auto")
        else:
            im = ax.imshow(data, origin="lower", extent=extent,
                           cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_xlabel("z [mm]", fontsize=fontsize)
        ax.set_ylabel("x [mm]", fontsize=fontsize)
        ax.set_title(title, fontsize=fontsize)
        cb = plt.colorbar(im, ax=ax, pad=0.02)
        cb.ax.tick_params(labelsize=6)
        if unit:
            cb.set_label(unit, fontsize=6)
        return im

    # ── Figure 1: Cardiac Waveform ─────────────────────────────────────────

    def plot_lambda_matrix(
        self,
        Lambda: np.ndarray,
        xv_mm: np.ndarray,
        iy: int,
        inside_r0: Optional[np.ndarray] = None,
        y_mm: Optional[float] = None,
    ):
        """
        Voxel-wise Λ(r) on one plane through the FOV.

        Two display points that materially affect how this figure reads:

        1. r > R0 voxels are MASKED (grey) when `inside_r0` is supplied.
           Outside the manufacturer's calibration sphere the radial factor
           of the SH expansion is clamped (see _bz_field), so Λ there is a
           hybrid of the clamped and unclamped regimes — not a field value.
           On the default 97³ grid spanning ±220 mm with R0 = 250 mm this is
           29.8 % of the volume, and the corner voxels reach r = 1.5·R0. Left
           unmasked they not only display meaningless numbers, they SET THE
           COLOUR SCALE (vm is the max over the plane), compressing the
           physically valid centre into the pale middle of the colourmap.

        2. The plane is labelled with its ACTUAL y position. On an even grid
           `linspace(-r_max, r_max, N)` contains no sample at y = 0, so the
           central index sits half a voxel off — 4.7 mm on the default grid.
           Set SimulationConfig.grid_N to an ODD number to place a sample
           exactly at isocentre.
        """
        extent = [xv_mm.min(), xv_mm.max(), xv_mm.min(), xv_mm.max()]
        ax_names = ["x", "y", "z"]
        m2d = None if inside_r0 is None else np.asarray(inside_r0)[:, iy, :]
        fig, axes = plt.subplots(3, 3, figsize=(12, 11))
        # The offset is deliberate, not an artefact of where the grid falls:
        # y = 0 is a symmetry plane on which four cross terms vanish (see
        # FIG_CORONAL_SLICE_MM). Say which it is, so a reader comparing this
        # figure with Markl 2003 Fig. 1 knows what plane they are looking at.
        _yl = ("y = 0" if y_mm is None else
               f"y = {y_mm:+.1f} mm"
               + (" (off the y = 0 symmetry plane, where Λ_xy, Λ_yx, Λ_yz "
                  "and Λ_zy vanish identically)" if abs(y_mm) > 1e-6 else
                  " (symmetry plane: Λ_xy, Λ_yx, Λ_yz, Λ_zy vanish here)"))
        fig.suptitle(
            f"Gradient Distortion Matrix  Λ(r)  —  {self.coil}\n"
            f"(coronal slice, {_yl})"
            + ("" if m2d is None else
               "   ·   grey = outside calibration sphere r > R₀ "
               "(extrapolated, excluded from the colour scale)"),
            fontsize=10, fontweight="bold")

        for j in range(3):
            for i in range(3):
                sl = Lambda[j, i, :, iy, :]
                ax = axes[j, i]
                dat = (sl - 1.0) if i == j else sl
                if m2d is not None:
                    dat = np.where(m2d, dat, np.nan)
                _fin = np.isfinite(dat)
                _amax = float(np.abs(dat[_fin]).max()) if _fin.any() else 0.0
                if i == j:
                    ttl  = (f"$\\Lambda_{{{ax_names[j]}{ax_names[i]}}} - 1$"
                            "  (self-coupling)")
                    vm   = max(_amax, 0.02)
                    _cm  = plt.get_cmap("RdBu_r").copy()
                else:
                    ttl  = (f"$\\Lambda_{{{ax_names[j]}{ax_names[i]}}}$"
                            "  (cross-coupling)")
                    vm   = max(_amax, 0.005)
                    _cm  = plt.get_cmap("PuOr").copy()
                _cm.set_bad(color="lightgrey")
                self._imshow_2d(ax, dat, extent, ttl,
                                _cm, vcenter=0, vmin=-vm, vmax=vm,
                                unit="relative deviation")
        fig.tight_layout()
        self._save(fig, "fig02_lambda_matrix.png")

    # ── Figure 3: Strength & angular deviation ─────────────────────────────

    def plot_pct_error(
        self,
        v_true: np.ndarray,
        v_meas: np.ndarray,
        v_cor:  np.ndarray,
        xv_mm:  np.ndarray,
        iy: int,
        t_sys: int,
        mode_label: str,
        inside_r0:  Optional[np.ndarray] = None,
        vessel_threshold_m_s: float = 0.05,
        snr_db: Optional[float] = None,
        steady: bool = False,
    ):
        """
        Percentage velocity-magnitude error before/after correction.
        `steady=True` (Mode 1 default) labels the frame as a steady field:
        every phase is identical, so this voxel set equals the time-averaged
        set of fig04.

        Notes:
        ------
        1. Vessel mask: voxels with |v_true| < vessel_threshold (default
           5 cm/s) are excluded. Using a near-zero threshold would divide
           by near-noise values and produce meaningless (very large)
           percentage errors that dominate the histogram, so a physical
           flow threshold is applied instead.
        2. Inside-R₀ mask: only voxels where the correction is valid.
        3. Colour scale is data-driven (95th percentile of the
           uncorrected error) so before/after contrast is visible.
        4. Map shows GREY for masked voxels (no signal there to evaluate).
        5. Improvement metric computed only on the valid vessel-mask voxels.

        Parameters
        ----------
        vessel_threshold_m_s : float, default 0.05
            Voxels with |v_true| below this value are excluded from the
            % error calculation. Physically: we only evaluate where there
            is actual flow signal to compare against.
        """
        # ── Compute velocity magnitudes ──────────────────────────────────
        mag_t = np.linalg.norm(v_true[..., t_sys], axis=0)   # (Nx,Ny,Nz)
        mag_s = np.linalg.norm(v_meas[..., t_sys], axis=0)
        mag_c = np.linalg.norm(v_cor [..., t_sys], axis=0)

        # ── Build the validity mask ──────────────────────────────────────
        # (a) vessel-signal mask: only voxels where v_true is large enough
        # that a percentage is physically meaningful
        vessel_mask = mag_t > vessel_threshold_m_s
        # (b) inside-R0 mask: only voxels where the correction is valid
        if inside_r0 is not None:
            valid_mask = vessel_mask & inside_r0
        else:
            valid_mask = vessel_mask

        n_valid = int(valid_mask.sum())
        n_total = valid_mask.size

        # ── Compute % error only at valid voxels ─────────────────────────
        # Set invalid voxels to NaN so they appear grey in the imshow.
        err_s = np.full_like(mag_t, np.nan, dtype=np.float64)
        err_c = np.full_like(mag_t, np.nan, dtype=np.float64)
        err_s[valid_mask] = 100.0 * np.abs(mag_s[valid_mask] - mag_t[valid_mask]) \
                                  / mag_t[valid_mask]
        err_c[valid_mask] = 100.0 * np.abs(mag_c[valid_mask] - mag_t[valid_mask]) \
                                  / mag_t[valid_mask]

        # ── Data-driven colour scale ───────────────────────────────────
        # Scale the colour range to the actual error distribution so the
        # before/after contrast is visible.  Use the 95th percentile of the
        # UNCORRECTED error (shared by both panels so they stay directly
        # comparable), with a sensible floor and ceiling.
        if np.any(valid_mask):
            emax = float(np.nanpercentile(err_s[valid_mask], 95))
            # Floor lowered from 3.0 to 1.0. With a near-isocentre tube the
            # 95th-percentile UNCORRECTED error is only ~1.5%, so a 3% floor
            # painted both panels in colour and made a good correction look
            # like a "red carpet". A 1% floor lets the corrected panel go
            # pale/white when the residual really is sub-percent, so the
            # before/after contrast is visible. The ceiling (50%) still
            # protects the scale on a strongly-distorted off-centre case.
            emax = float(np.clip(emax, 1.0, 50.0))
        else:
            emax = 50.0

        # ── Build figure ────────────────────────────────────────────────
        extent = [xv_mm.min(), xv_mm.max(), xv_mm.min(), xv_mm.max()]
        cmap   = plt.cm.hot_r.copy()
        cmap.set_bad(color="lightgrey")     # NaN → grey

        fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
        # The SNR is stamped into the title (as fig04 already does) because the
        # % improvement shown here is a NOISE statistic: it is only meaningful
        # alongside the SNR it was computed at. A high simulation SNR is chosen
        # deliberately so the SYSTEMATIC, spatially structured GNL error stays
        # visible; the noise dependence belongs to the SNR sweep table and the
        # SNR-independent noise-free row.
        _snr_line = ("" if snr_db is None else
                     f"\nsimulated at SNR = {snr_db:.0f} dB "
                     "(the % improvement is a noise statistic — "
                     "see the SNR sweep)")
        fig.suptitle(
            f"Velocity magnitude error before and after correction — "
            f"{mode_label}{_snr_line}",
            fontsize=11, fontweight="bold")

        # Panel 1 — standard recon
        #
        # AXIS ORIENTATION — do not reintroduce a .T here.
        # err_s[:, iy, :] has shape (Nx, Nz). imshow maps ROWS to the VERTICAL
        # axis and COLUMNS to the HORIZONTAL axis, so this array gives
        # vertical = x, horizontal = z, which is what the labels below say and
        # what _imshow_2d (fig01, fig03) produces for the same coronal plane.
        # A previous version passed .T, silently swapping the two: the vessel
        # is built with NO x dependence (Gaussian in rho = sqrt(y^2+z^2)), so
        # it spans the full +/-220 mm in x and only +/-14 mm in z — and in the
        # transposed panel that band ran along the axis labelled "z".
        im1 = axes[0].imshow(err_s[:, iy, :], origin="lower", extent=extent,
                             cmap=cmap, vmin=0, vmax=emax, aspect="auto")
        axes[0].set_title("Standard recon (no correction)\n"
                          f"{n_valid:,}/{n_total:,} voxels evaluated "
                          + ("(steady field)" if steady
                             else "(peak-systole phase)"), fontsize=9)
        axes[0].set_xlabel("z [mm]"); axes[0].set_ylabel("x [mm]")
        cb1 = plt.colorbar(im1, ax=axes[0], pad=0.02)
        cb1.set_label(f"%  (0-{emax:.0f})", fontsize=7); cb1.ax.tick_params(labelsize=6)

        # Panel 2 — corrected
        im2 = axes[1].imshow(err_c[:, iy, :], origin="lower", extent=extent,
                             cmap=cmap, vmin=0, vmax=emax, aspect="auto")
        axes[1].set_title("Generalised recon (corrected)", fontsize=9)
        axes[1].set_xlabel("z [mm]"); axes[1].set_ylabel("x [mm]")
        cb2 = plt.colorbar(im2, ax=axes[1], pad=0.02)
        cb2.set_label(f"%  (0-{emax:.0f})", fontsize=7); cb2.ax.tick_params(labelsize=6)

        # Panel 3 — histogram (vessel-mask voxels only)
        flat_s = err_s[valid_mask]
        flat_c = err_c[valid_mask]
        if flat_s.size > 0:
            hist_max = float(min(np.percentile(flat_s, 99), 80.0))
            bins  = np.linspace(0, max(hist_max, 5.0), 41)
            med_s = float(np.median(flat_s))
            med_c = float(np.median(flat_c))
            improvement = (1.0 - med_c / max(med_s, 1e-6)) * 100.0

            axes[2].hist(flat_s, bins=bins, alpha=0.72,
                         color=PALETTE["std"],
                         label=f"Standard  (median {med_s:.1f}%)")
            axes[2].hist(flat_c, bins=bins, alpha=0.72,
                         color=PALETTE["corrected"],
                         label=f"Corrected (median {med_c:.2f}%)")
            axes[2].set_xlabel("% velocity error  (vessel voxels only)")
            axes[2].set_ylabel("Voxel count")
            axes[2].set_title(
                "Error histogram\n"
                f"({n_valid:,} voxels, |v_true| > "
                f"{vessel_threshold_m_s*100:.0f} cm/s"
                + (" — steady flow: same set as fig04)" if steady else
                   " at PEAK SYSTOLE — a different set from fig04, "
                   "which uses the time average)"),
                fontsize=8)
            axes[2].legend(fontsize=8, loc="upper right")
            axes[2].grid(axis="y", alpha=0.3)
            axes[2].text(0.95, 0.78,
                         f"Improvement: {improvement:+.1f}%",
                         transform=axes[2].transAxes, ha="right",
                         fontsize=10, fontweight="bold",
                         color=(PALETTE["corrected"] if improvement > 0 else "#A32D2D"),
                         bbox=dict(facecolor="white", alpha=0.85,
                                   edgecolor="lightgrey"))
        else:
            axes[2].text(0.5, 0.5,
                         "No voxels exceed\n"
                         f"|v_true| > {vessel_threshold_m_s*100:.2g} cm/s\n"
                         "and r ≤ R₀\n"
                         "(no vessel in this slice)",
                         transform=axes[2].transAxes,
                         ha="center", va="center", fontsize=10,
                         color="grey")
            axes[2].set_xticks([]); axes[2].set_yticks([])
            axes[2].set_title("Error histogram (no valid voxels)", fontsize=9)

        fig.tight_layout()
        self._save(fig, "fig07_pct_error.png")

    # ── Figure 8: Radial stratification ───────────────────────────────────

    def plot_spatially_varying_venc(
        self,
        Lambda:        np.ndarray,
        VENC_nominal:  float,
        xv_mm:         np.ndarray,
        iy:            int,
        inside_r0:     Optional[np.ndarray] = None,
        y_mm:          Optional[float] = None,
    ):
        """
        Compute and visualise the spatially varying VENC field that arises
        from gradient nonlinearity :

            VENC(r) = VENC_nominal / ‖Λ(:, i)(r)‖

        where ‖Λ(:, i)‖ is the per-encoding-axis strength factor. This
        figure addresses : the spatially varying VENC is
        the UNIQUE signature of gradient nonlinearity. Additive phase
        offsets (eddy currents, concomitant fields — out of scope here)
        shift velocities by a constant; they cannot change VENC itself.
        Only a gradient-field deviation rescales the encoding.

        Why this matters
        --------------------------------
        Reviewers of will look for evidence
        that the simulation actually computes the quantity introduced in
  This figure makes that explicit:
          · Per-axis VENC maps for x, y, z encoding directions
          · Histogram of relative VENC deviation across the FOV
          · Percent-deviation summary statistics (median, p95, p99)
        """
        # Strength factor per axis: d_i = ‖Λ[:, i]‖
        d_i = np.sqrt(np.sum(Lambda ** 2, axis=0))         # (3, Nx, Ny, Nz)
        # Effective VENC per axis (m/s) → cm/s for display
        venc_r = (VENC_nominal / np.maximum(d_i, 1e-9)) * 100.0
        venc_nominal_cm = VENC_nominal * 100.0
        # Relative deviation (%) from nominal
        rel_dev = (venc_r - venc_nominal_cm) / venc_nominal_cm * 100.0

        extent   = [xv_mm.min(), xv_mm.max(), xv_mm.min(), xv_mm.max()]
        ax_names = ["x", "y", "z"]

        # Voxels beyond the calibration sphere are clamped extrapolations,
        # not field values (see plot_lambda_matrix). They are masked from the
        # maps AND excluded from the histogram and the quoted percentiles —
        # on the default grid they are 31% of the volume, so pooling them in
        # silently biases every number in the annotation box.
        _m3d = None if inside_r0 is None else np.asarray(inside_r0, bool)
        m2d  = None if _m3d is None else _m3d[:, iy, :]

        fig = plt.figure(figsize=(13, 8))
        gs  = gridspec.GridSpec(2, 3, height_ratios=[1.0, 0.85],
                                hspace=0.35, wspace=0.30)
        _yl = "" if y_mm is None else f",   coronal slice y = {y_mm:+.1f} mm"
        fig.suptitle(
            f"Spatially varying VENC(r) — {self.coil}\n"
            f"VENC(r) = VENC_nom / ‖Λ(:,i)(r)‖,   nominal "
            f"{venc_nominal_cm:.0f} cm/s{_yl}",
            fontsize=11, fontweight="bold")

        # Top row: per-axis VENC(r) maps
        for ci in range(3):
            ax  = fig.add_subplot(gs[0, ci])
            sl  = venc_r[ci, :, iy, :]
            if m2d is not None:
                sl = np.where(m2d, sl, np.nan)
            _dev = np.abs(sl - venc_nominal_cm)
            vm  = (float(np.nanpercentile(_dev, 98))
                   if np.any(np.isfinite(_dev)) else 1.0)
            # VENC = VENC_nom/‖Λ(:,i)‖ is strictly POSITIVE. Clamp the
            # lower display bound to 0 so the diverging colourbar never shows
            # physically impossible negative VENC (the z-axis deviation can be
            # ~480 %, which would drive vmin to ~−300 cm/s — a misleading
            # scale even though no voxel is actually negative).
            _cm = plt.get_cmap("RdBu_r").copy()
            _cm.set_bad(color="lightgrey")
            self._imshow_2d(ax, sl, extent,
                            f"VENC$_{{{ax_names[ci]}}}$(r)  —  "
                            f"{ax_names[ci]}-encoding",
                            _cm,
                            vcenter=venc_nominal_cm,
                            vmin=max(venc_nominal_cm - vm, 0.0),
                            vmax=venc_nominal_cm + vm,
                            unit="cm/s")

        # Bottom row: histogram of relative deviation
        ax_hist = fig.add_subplot(gs[1, :])
        # Per-axis samples, restricted to r ≤ R₀ when the mask is available.
        _rel = [rel_dev[ci][_m3d] if _m3d is not None else rel_dev[ci].ravel()
                for ci in range(3)]
        all_rel = np.concatenate(_rel)
        # Clip extreme tails for histogram readability
        clip    = float(np.percentile(np.abs(all_rel), 99.5))
        bins    = np.linspace(-clip, clip, 80)

        for ci, col in zip(range(3),
                            [PALETTE["highlight"], PALETTE["accent_amber"],
                             PALETTE["corrected"]]):
            ax_hist.hist(_rel[ci], bins=bins, alpha=0.55,
                         color=col, label=f"{ax_names[ci]}-encoding",
                         edgecolor="white", lw=0.2)
        ax_hist.axvline(0, color="k", lw=0.8, ls=":")
        ax_hist.set_xlabel("Relative VENC deviation  (VENC(r) − VENC_nom) / VENC_nom  [%]")
        ax_hist.set_ylabel("Voxel count")
        ax_hist.set_title("Distribution of spatially varying VENC deviation across the FOV")
        ax_hist.legend(fontsize=8)
        ax_hist.grid(axis="y", alpha=0.4)

        # Annotation: per-axis percentiles
        lines = ["VENC deviation  (% from nominal)"
                 + ("  —  r ≤ R₀ only" if _m3d is not None
                    else "  —  WHOLE GRID incl. extrapolated r > R₀")]
        for ci in range(3):
            _a  = np.abs(_rel[ci])
            med = float(np.median(_a))
            p95 = float(np.percentile(_a, 95))
            p99 = float(np.percentile(_a, 99))
            lines.append(
                f"  {ax_names[ci]}:  median |Δ| = {med:.2f}%   "
                f"95%-ile = {p95:.2f}%   99%-ile = {p99:.2f}%"
            )
        ax_hist.text(0.99, 0.97, "\n".join(lines),
                     transform=ax_hist.transAxes,
                     va="top", ha="right", fontsize=7,
                     family="monospace",
                     bbox=dict(facecolor="white", alpha=0.85,
                               edgecolor="gray"))

        self._save(fig, "fig17_spatially_varying_venc.png")

    # ── Figure 18: Correction operator structures ─────────

    def plot_summary_metrics(
        self,
        results: ValidationResults,
        mode_label: str,
    ):
        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
        nv = getattr(results, "vessel_n", 0)
        _snr_lbl = getattr(self, "snr_db_label", None)
        _thr     = getattr(self, "vessel_thresh_label", "5 cm/s")
        # The percentage improvement below is a noise statistic as much as a
        # correction statistic, so the threshold and the SNR it assumed stay
        # on the figure. Everything else that used to be here is in the
        # console output and run_config.json.
        fig.suptitle(
            f"Validation metrics — {mode_label}\n"
            f"vessel voxels, n = {nv:,}  (|v_true| > {_thr}, time average)"
            + ("" if _snr_lbl is None else f",  SNR = {_snr_lbl}"),
            fontsize=11, fontweight="bold")

        comp_names = ["|v|"]
        x = np.arange(len(comp_names))
        w = 0.35

        # Use vessel-masked scalar metrics (consistent with fig07/08/09).
        # Fall back to global metrics if vessel metrics are unavailable.
        rmse_b = getattr(results, "rmse_before_vessel", None)
        rmse_a = getattr(results, "rmse_after_vessel",  None)
        mae_b  = getattr(results, "mae_before_vessel",  None)
        mae_a  = getattr(results, "mae_after_vessel",   None)
        if rmse_b is None or rmse_b == 0.0:
            rmse_b = float(np.mean(results.rmse_before))
            rmse_a = float(np.mean(results.rmse_after))
            mae_b  = float(np.mean(results.mae_before))
            mae_a  = float(np.mean(results.mae_after))

        # RMSE
        axes[0].bar(x - w/2, [rmse_b*100], w, label="Before",
                    color=PALETTE["std"],       alpha=0.75)
        axes[0].bar(x + w/2, [rmse_a*100], w, label="After",
                    color=PALETTE["corrected"], alpha=0.75)
        axes[0].set_xticks(x); axes[0].set_xticklabels(comp_names)
        axes[0].set_ylabel("RMSE [cm/s]")
        axes[0].set_title("Root Mean Square Error")
        axes[0].legend(fontsize=7)
        axes[0].grid(axis="y")
        for xi, val in ((x[0] - w/2, rmse_b*100), (x[0] + w/2, rmse_a*100)):
            axes[0].text(float(xi), val, f"{val:.2f}", ha="center",
                         va="bottom", fontsize=8)
        # Annotate improvement
        if rmse_b > 0:
            impr = 100.0 * (rmse_b - rmse_a) / rmse_b
            axes[0].text(0.5, 0.92, f"{impr:+.0f}%",
                         transform=axes[0].transAxes, ha="center",
                         fontsize=11, fontweight="bold",
                         color=(PALETTE["corrected"] if impr > 0
                                else PALETTE["std"]))

        # MAE
        axes[1].bar(x - w/2, [mae_b*100], w, label="Before",
                    color=PALETTE["std"],       alpha=0.75)
        axes[1].bar(x + w/2, [mae_a*100], w, label="After",
                    color=PALETTE["corrected"], alpha=0.75)
        axes[1].set_xticks(x); axes[1].set_xticklabels(comp_names)
        axes[1].set_ylabel("MAE [cm/s]")
        axes[1].set_title("Mean Absolute Error")
        axes[1].legend(fontsize=7)
        axes[1].grid(axis="y")
        for xi, val in ((x[0] - w/2, mae_b*100), (x[0] + w/2, mae_a*100)):
            axes[1].text(float(xi), val, f"{val:.2f}", ha="center",
                         va="bottom", fontsize=8)
        if mae_b > 0:
            impr = 100.0 * (mae_b - mae_a) / mae_b
            axes[1].text(0.5, 0.92, f"{impr:+.0f}%",
                         transform=axes[1].transAxes, ha="center",
                         fontsize=11, fontweight="bold",
                         color=(PALETTE["corrected"] if impr > 0
                                else PALETTE["std"]))

        # p95 velocity error (worst-case, vessel-masked). A magnitude metric
        # like RMSE/MAE that reliably shows the correction's benefit. Pearson r
        # is intentionally NOT shown: it stays ~1.0 both before and after because
        # gradient nonlinearity is a smooth multiplicative distortion that
        # preserves velocity ranking, so it cannot discriminate here. RMSE, MAE
        # and p95 error are the discriminating metrics.
        p95_b = float(getattr(results, "p95_before_vessel", 0.0)) * 100.0
        p95_a = float(getattr(results, "p95_after_vessel",  0.0)) * 100.0
        axes[2].bar(x - w/2, [p95_b], w, label="Before",
                    color=PALETTE["std"],       alpha=0.75)
        axes[2].bar(x + w/2, [p95_a], w, label="After",
                    color=PALETTE["corrected"], alpha=0.75)
        axes[2].set_xticks(x); axes[2].set_xticklabels(comp_names)
        axes[2].set_ylabel("p95 error [cm/s]")
        axes[2].set_title("95th-percentile Error (worst-case)")
        axes[2].legend(fontsize=7)
        axes[2].grid(axis="y")
        for xi, val in ((x[0] - w/2, p95_b), (x[0] + w/2, p95_a)):
            axes[2].text(float(xi), val, f"{val:.2f}", ha="center",
                         va="bottom", fontsize=8)
        if p95_b > 0:
            impr = 100.0 * (p95_b - p95_a) / p95_b
            axes[2].text(0.5, 0.92, f"{impr:+.0f}%",
                         transform=axes[2].transAxes, ha="center",
                         fontsize=11, fontweight="bold",
                         color=(PALETTE["corrected"] if impr > 0 else PALETTE["std"]))

        fig.tight_layout()
        self._save(fig, "fig12_summary_metrics.png")


# ═════════════════════════════════════════════════════════════════════════════
# MAIN ORCHESTRATION
# ═════════════════════════════════════════════════════════════════════════════

def run_simulation_mode(
    mode: int,
    X: np.ndarray, Y: np.ndarray, Z: np.ndarray,
    Lambda: np.ndarray,
    d_i: np.ndarray,
    theta: np.ndarray,
    xv_mm: np.ndarray,
    cfg: SimulationConfig,
    grad_coeffs: GradientCoefficients,
    plotter: PlottingEngine,
    val_engine: ValidationEngine,
    corrector:  VelocityCorrector,
    inside_r0:  Optional[np.ndarray] = None,
):
    """Run a single simulation mode and generate all outputs.

    The optional `inside_r0` boolean mask (shape Nx,Ny,Nz) restricts all
    quantitative validation metrics to voxels inside the spherical-harmonic
    calibration sphere. The expansion coefficients are only defined on that
    sphere (Romeo & Hoult 1984; Markl et al. 2003, Appendix Eq. [A1]), so
    metrics computed outside it would report a clamp artefact rather than a
    measurement — see GradientFieldEvaluator.inside_r0_mask.
    """
    sim     = SimulationEngine4D(cfg)
    wave    = sim.cardiac_waveform()
    N       = cfg.grid_N
    iy      = N // 2
    t_sys   = int(np.argmax(wave))

    # The label is built AFTER the field is generated, because
    # generate_poiseuille_4d computes the flow rate from the configured peak
    # velocity and tube radius. The previous hard-coded "Q = 300 ml/min"
    # was a stale default and understated the simulated flow by ~226x.
    if mode == 1:
        v_true = sim.generate_poiseuille_4d(X, Y, Z)
        # Q is still computed, printed and stored in run_config.json; it is
        # left out of the title because the enlarged sim tube makes it
        # non-physiological (tens of l/min) and it distracts from the
        # velocity field that is actually being verified.
        label  = ("Mode 1 — Poiseuille Tube Phantom  "
                  f"(v_max = {cfg.poiseuille_v_peak_m_s*100:.0f} cm/s, "
                  + ("mildly pulsatile 85–100 %)"
                     if getattr(cfg, "mode1_pulsatile", False)
                     else "steady)"))
    else:
        v_true = sim.generate_pulsatile_vessel_4d(X, Y, Z)
        # NAMING (see M3). This is a Gaussian tube of FWHM = cfg.vessel_fwhm_m
        # running the FULL length of the FOV through isocentre. At the default
        # 25 mm FWHM it is aorta-scale, not cerebrovascular — intracranial
        # arteries are 2-4 mm and would be far below one voxel on this grid.
        # The title states the geometry instead of implying an anatomy the
        # simulation does not have.
        label  = ("Mode 2 — Pulsatile Tube Phantom  "
                  f"(FWHM {cfg.vessel_fwhm_m*1000:.0f} mm, "
                  f"{cfg.flow_pattern}, {cfg.Nt} phases)")

    print(f"\n{'='*60}\n  {label}\n{'='*60}")

    # Vessel-signal threshold as a fraction of peak velocity (no fixed cap).
    # This excludes the low-velocity vessel wall and ends, where the % error
    # 100*|dv|/|v_true| is a divide-by-small artefact rather than residual
    # distortion. It scales with the flow regime: for Mode 2 (peak ~80 cm/s)
    # it is ~16 cm/s; for Mode 1 Poiseuille it likewise tracks the configured
    # peak, so the vessel mask is never empty. Tune via
    # PCT_ERROR_VESSEL_FRACTION.
    v_peak_true   = float(np.linalg.norm(v_true, axis=0).max())
    VESSEL_THRESH = max(PCT_ERROR_VESSEL_FRACTION * v_peak_true, 1e-4)   # m/s

    # ── Discretisation honesty (see M3) ──────────────────────────────────
    # The peak of a Gaussian/parabolic profile is under-read on a coarse grid,
    # and an EVEN grid_N has no sample at rho = 0 at all, so the on-grid
    # maximum is lower than the configured peak. Every percentage error in
    # fig02 is referenced to this partial-volumed truth, so the shortfall is
    # measured and reported rather than left implicit.
    _v_cfg = float(cfg.v_peak_systole if mode == 2 else cfg.poiseuille_v_peak_m_s)
    _vox_mm_r = float(cfg.r_max_m * 2000.0 / max(cfg.grid_N - 1, 1))
    _feature_mm = float(cfg.vessel_fwhm_m * 1000.0 if mode == 2
                        else cfg.tube_radius_m * 2000.0)
    _vpf = _feature_mm / _vox_mm_r if _vox_mm_r > 0 else 0.0
    if _v_cfg > 0:
        _short = 100.0 * (1.0 - v_peak_true / _v_cfg)
        print(f"  [grid] Peak |v_true| on the grid = {v_peak_true*100:.2f} "
              f"cm/s vs configured {_v_cfg*100:.0f} cm/s "
              f"({_short:+.1f}% partial-volume shortfall); "
              f"{_vpf:.1f} voxels across the "
              f"{'FWHM' if mode == 2 else 'tube diameter'}.")
        if _short > 5.0:
            print("         All % errors are referenced to this under-read "
                  "peak. Raise cfg.grid_N or lower")
            print("         cfg.r_max_m before quoting absolute velocities "
                  "from this mode.")
    # Make the assumed noise level visible on every figure the plotter draws.
    plotter.snr_db_label = f"{cfg.snr_db:.0f} dB" if cfg.add_noise else "noise off"
    print(f"  [mask] Vessel threshold = {VESSEL_THRESH*100:.2f} cm/s "
          f"({PCT_ERROR_VESSEL_FRACTION*100:.0f}% of peak {v_peak_true*100:.1f} cm/s); "
          "low-velocity wall/end voxels excluded from % error and vessel metrics.")
    # Expose the actual threshold so every figure/label reports the TRUE
    # value used (e.g. 0.07 cm/s for Mode 1), not a hardcoded "5 cm/s".
    VTHR_CM = VESSEL_THRESH * 100.0
    VTHR_LABEL = f"{VTHR_CM:.2g} cm/s"
    plotter.vessel_thresh_label = VTHR_LABEL

    # ── Forward distortion ────────────────────────────────────────────────
    print("\n  [pipeline] Step 0: Forward distortion  Λ(r) · v_true …")
    v_meas = SimulationEngine4D.apply_forward_distortion(v_true, Lambda)

    # ── Add noise ─────────────────────────────────────────────────────────
    if cfg.add_noise:
        print("  [pipeline] Step 0b: Injecting realistic phase noise …")
        # Use a deterministic per-mode RNG so Mode 1 and Mode 2
        # in the same run get reproducible — and independent — noise fields.
        noise_rng = np.random.default_rng(seed=42 + mode)
        v_meas = sim.add_phase_noise(v_meas, cfg.VENC, rng=noise_rng)

    # ── GNL correction ────────────────────────────────────────────────────
    print("  [pipeline] Step 3: Gradient nonlinearity correction  Λ(r)⁻¹ …")
    t0    = time.perf_counter()
    v_cor = corrector.correct_4d(v_meas, Lambda)
    print(f"    Correction time: {time.perf_counter() - t0:.2f} s")

    # ── Metrics ───────────────────────────────────────────────────────────
    # Metrics are restricted to voxels inside the spherical-harmonic
    # calibration sphere r ≤ R0 when the mask is given.
    print("\n  [metrics] Computing validation metrics …")
    if inside_r0 is not None:
        n_in = int(inside_r0.sum())
        n_tot = inside_r0.size
        print(f"  [metrics] Restricted to {n_in}/{n_tot} voxels inside "
              f"calibration sphere ({100*n_in/n_tot:.1f}%).")
        if n_in < 0.9 * n_tot:
            print(f"  [metrics] NOTE: {100*(1-n_in/n_tot):.1f}% of the grid "
                  f"lies outside R0 = {grad_coeffs.R0*1000:.0f} mm, where the "
                  "radial clamp forces Lambda -> I. Those voxels are excluded "
                  "from every metric. To keep the whole grid inside the "
                  "calibration sphere set cfg.r_max_m <= R0/sqrt(3) = "
                  f"{grad_coeffs.R0/np.sqrt(3):.3f} m.")

    # Average over time for spatial metrics
    vt_avg = v_true.mean(axis=-1)
    vm_avg = v_meas.mean(axis=-1)
    vc_avg = v_cor.mean(axis=-1)

    results = ValidationResults(
        rmse_before = val_engine.rmse(vm_avg, vt_avg, mask=inside_r0),
        rmse_after  = val_engine.rmse(vc_avg, vt_avg, mask=inside_r0),
        mae_before  = val_engine.mae(vm_avg,  vt_avg, mask=inside_r0),
        mae_after   = val_engine.mae(vc_avg,  vt_avg, mask=inside_r0),
    )
    # rmse_before/after above are WHOLE-SPHERE diagnostics.
    # The headline pct_improvement is set later from the vessel-region
    # metrics (see below) so console/JSON match figs 07/08/12/13.
    results.rmse_before_whole = results.rmse_before
    results.rmse_after_whole  = results.rmse_after
    results.pct_improvement_whole = val_engine.pct_improvement(
        results.rmse_before, results.rmse_after)

    pb = val_engine.pearson(vm_avg, vt_avg, mask=inside_r0)
    pa = val_engine.pearson(vc_avg, vt_avg, mask=inside_r0)
    results.pearson_before = np.array(pb)
    results.pearson_after  = np.array(pa)

    # ── Vessel-masked metrics for fig12 summary ───────────────────────────
    # The global metrics above (over all R0 voxels) are dominated by
    # empty-tissue voxels where Λ⁻¹ amplifies noise, which makes the
    # correction look worse than it is. For the fig12 scoreboard we
    # additionally compute metrics on vessel voxels only, using the same
    # run-time VESSEL_THRESH as the % error map, so the summary reflects
    # correction performance on voxels carrying real flow signal.
    vt_mag_avg  = np.linalg.norm(vt_avg, axis=0)
    vessel_m    = (vt_mag_avg > VESSEL_THRESH)
    if inside_r0 is not None:
        vessel_m = vessel_m & inside_r0
    results.vessel_n          = int(vessel_m.sum())
    # Carry the real threshold out of this function so main()'s summary can
    # report the value actually used instead of a hardcoded placeholder.
    results.vessel_thresh_label = VTHR_LABEL
    results.rmse_before_vessel = val_engine.rmse(vm_avg, vt_avg, mask=vessel_m)
    results.rmse_after_vessel  = val_engine.rmse(vc_avg, vt_avg, mask=vessel_m)
    results.mae_before_vessel  = val_engine.mae(vm_avg,  vt_avg, mask=vessel_m)
    results.mae_after_vessel   = val_engine.mae(vc_avg,  vt_avg, mask=vessel_m)
    results.pearson_before_vessel = np.array(
        val_engine.pearson(vm_avg, vt_avg, mask=vessel_m))
    results.pearson_after_vessel  = np.array(
        val_engine.pearson(vc_avg, vt_avg, mask=vessel_m))
    # p95 velocity error (95th percentile of |v_est - v_true|) on vessel voxels
    _errb = np.linalg.norm(vm_avg - vt_avg, axis=0)[vessel_m]
    _erra = np.linalg.norm(vc_avg - vt_avg, axis=0)[vessel_m]
    results.p95_before_vessel = float(np.percentile(_errb, 95)) if _errb.size else 0.0
    results.p95_after_vessel  = float(np.percentile(_erra, 95)) if _erra.size else 0.0
    print(f"  {_fig_label(12)} Vessel-masked summary metrics on "
          f"{results.vessel_n:,} voxels (|v_true| > {VTHR_LABEL}):")
    print(f"          RMSE {results.rmse_before_vessel*100:.3f} → "
          f"{results.rmse_after_vessel*100:.3f} cm/s  |  "
          f"MAE {results.mae_before_vessel*100:.3f} → "
          f"{results.mae_after_vessel*100:.3f} cm/s")

    # Headline improvement = vessel region (matches the figures). Assigned
    # here, before the three-way comparison below prints it.
    results.pct_improvement = val_engine.pct_improvement(
        results.rmse_before_vessel, results.rmse_after_vessel)

    # ── Separate the SYSTEMATIC error from the NOISE ─────────────────────
    # See the ValidationResults docstring. Three numbers, not one:
    #   (a) time-averaged   — the historical headline, noise suppressed by √Nt
    #   (b) per frame       — the same metric with no temporal averaging
    #   (c) noise-free      — the run repeated with noise off: the pure GNL
    #                         error and the pure residual, independent of SNR
    _e_b_pf = np.sqrt(np.mean(np.sum((v_meas - v_true) ** 2, axis=0)[vessel_m]))
    _e_a_pf = np.sqrt(np.mean(np.sum((v_cor  - v_true) ** 2, axis=0)[vessel_m]))
    results.rmse_before_vessel_perframe = float(_e_b_pf)
    results.rmse_after_vessel_perframe  = float(_e_a_pf)
    results.pct_improvement_perframe = val_engine.pct_improvement(
        float(_e_b_pf), float(_e_a_pf))
    results.snr_db = float(cfg.snr_db) if cfg.add_noise else float("nan")
    results.v_peak_configured_m_s = float(_v_cfg)
    results.v_peak_on_grid_m_s    = float(v_peak_true)
    results.voxels_per_fwhm       = float(_vpf)

    # Noise-free repeat. Costs one extra forward + one extra inversion; on the
    # default grid that is a couple of seconds and it is the only number here
    # that characterises the CORRECTION rather than the assumed SNR.
    _vm_clean = SimulationEngine4D.apply_forward_distortion(v_true, Lambda)
    _vc_clean = corrector.correct_4d(_vm_clean, Lambda)
    _sys_b = val_engine.rmse(_vm_clean.mean(axis=-1), vt_avg, mask=vessel_m)
    _sys_a = val_engine.rmse(_vc_clean.mean(axis=-1), vt_avg, mask=vessel_m)
    results.rmse_systematic_before_vessel = float(_sys_b)
    results.rmse_systematic_after_vessel  = float(_sys_a)
    results.pct_improvement_systematic = val_engine.pct_improvement(
        float(_sys_b), float(_sys_a))
    del _vm_clean, _vc_clean

    print("          ── the same comparison, three ways ──")
    print(f"          time-averaged : {results.rmse_before_vessel*100:.4f} → "
          f"{results.rmse_after_vessel*100:.4f} cm/s  "
          f"({results.pct_improvement:+.1f}%)   "
          f"[noise suppressed by √Nt = √{cfg.Nt}]")
    print(f"          per frame     : {_e_b_pf*100:.4f} → {_e_a_pf*100:.4f} "
          f"cm/s  ({results.pct_improvement_perframe:+.1f}%)   "
          f"[no temporal averaging]")
    print(f"          noise-free    : {_sys_b*100:.4f} → {_sys_a*100:.3e} "
          f"cm/s  ({results.pct_improvement_systematic:+.4f}%)   "
          "[pure systematic GNL error — SNR-independent]")
    print("          Quote the NOISE-FREE row as the correction result. The "
          "other two are")
    print(f"          noise statistics at the assumed SNR = {cfg.snr_db:.0f} "
          "dB and change with it.")

    # Bootstrap 95 % confidence intervals on RMSE ─────────
    # Headline CI uses the SAME vessel mask as the headline
    # pct_improvement; a whole-sphere CI is also computed as a diagnostic.
    print("\n  [stats] Bootstrap 95 % confidence intervals on RMSE …")
    rmse_b_pt, b_lo, b_hi = val_engine.rmse_bootstrap_ci(
        vm_avg, vt_avg, mask=vessel_m)
    rmse_a_pt, a_lo, a_hi = val_engine.rmse_bootstrap_ci(
        vc_avg, vt_avg, mask=vessel_m)
    results.rmse_before_ci = (b_lo, b_hi)
    results.rmse_after_ci  = (a_lo, a_hi)
    # Whole-sphere diagnostic CI
    _, bw_lo, bw_hi = val_engine.rmse_bootstrap_ci(
        vm_avg, vt_avg, mask=inside_r0)
    _, aw_lo, aw_hi = val_engine.rmse_bootstrap_ci(
        vc_avg, vt_avg, mask=inside_r0)
    results.rmse_before_ci_whole = (bw_lo, bw_hi)
    results.rmse_after_ci_whole  = (aw_lo, aw_hi)
    # ── Paired improvement CI (the significance test to quote) ──────────
    # Before/after errors are computed on the SAME voxels, so the
    # improvement is bootstrapped as a PAIRED statistic: each replicate
    # resamples once and evaluates Δ_b and I_b inside the replicate.
    # Whole x-slices are resampled as spatial blocks rather than voxels —
    # neighbouring voxels share the smooth Λ field and noise structure, so
    # voxel-wise resampling overstates the effective N and narrows the CI.
    # The separate before/after CIs above remain as descriptive intervals;
    # their non-overlap was the previous significance rule (conservative
    # but low-powered, and the old improvement interval built from opposing
    # CI endpoints was a worst-case bound, not a CI).
    _blk_ix = np.broadcast_to(
        np.arange(X.shape[0])[:, None, None], X.shape)[vessel_m]
    (_impr_pt, pct_lo, pct_hi,
     _d_lo, _d_hi) = val_engine.paired_improvement_bootstrap_ci(
        vm_avg, vc_avg, vt_avg, mask=vessel_m, block_index=_blk_ix)
    results.pct_improvement_ci = (pct_lo, pct_hi)
    # Significance: the paired improvement CI excludes zero
    significant = pct_lo > 0.0

    # Flow rate (Mode 1) — recompute tube_mask with current cfg radius
    # Voxel spacing from the ACTUAL coordinate arrays. The grid is built by
    # linspace(-r_max, r_max, N), whose step is 2*r_max/(N-1); the previous
    # expression 2*r_max/N under-sized every voxel and biased the integrated
    # flow rate low by a factor ((N-1)/N)^2 — about -2.1% at the default
    # N = 97 (-4.1% at N = 48). Never reconstruct spacing independently of
    # the grid that defines it (the partial-volume check already used the
    # correct (N-1) form; the two now agree).
    dy_grid     = float(abs(Y[0, 1, 0] - Y[0, 0, 0]))
    dz_grid     = float(abs(Z[0, 0, 1] - Z[0, 0, 0]))
    voxel_size  = dy_grid
    voxel_area  = dy_grid * dz_grid      # flux plane ⟂ x carries v_x
    rho         = np.sqrt(Y**2 + Z**2)
    tube_mask   = rho < cfg.tube_radius_m
    n_tube_vox  = tube_mask.sum()
    if n_tube_vox == 0:
        print("  [WARN] tube_mask is empty — tube_radius "
              f"({cfg.tube_radius_m*1000:.0f} mm) < voxel_size "
              f"({voxel_size*1000:.1f} mm). Flow rate will be 0.")
    else:
        print(f"  [flow] Tube cross-section: {n_tube_vox} voxels  "
              f"(r={cfg.tube_radius_m*1000:.0f} mm, "
              f"voxel={voxel_size*1000:.1f} mm)")
    results.flow_rate_ref  = cfg.flow_rate_ml_min
    results.flow_rate_std  = val_engine.flow_rate_ml_min(v_meas, tube_mask, voxel_area)
    results.flow_rate_cor  = val_engine.flow_rate_ml_min(v_cor,  tube_mask, voxel_area)

    # NOTE: this mask is the TIME-AVERAGED |v_true|, the same one fig04 uses.
    # fig02 masks on the peak-systole phase. For a pulsatile field its voxel
    # count is therefore larger; for steady Mode 1 the two sets are identical.
    print(f"\n  ┌─ RESULTS  (vessel region, time-avg |v_true|>{VTHR_LABEL} "
          f"— same mask as {_fig_list(12, 13)}) ┐")
    print(f"  │  Metric basis: TIME-AVERAGED field, SNR = "
          f"{cfg.snr_db:.0f} dB{'' if cfg.add_noise else ' (noise off)'}. "
          "Both choices inflate the % below —")
    print("  │    see the noise-free row for the SNR-independent result.")
    print(f"  │  RMSE before : {results.rmse_before_vessel*100:.3f} cm/s  "
          f"95% CI [{b_lo*100:.3f}, {b_hi*100:.3f}]")
    print(f"  │  RMSE after  : {results.rmse_after_vessel*100:.3f} cm/s  "
          f"95% CI [{a_lo*100:.3f}, {a_hi*100:.3f}]")
    print(f"  │  Improvement : {results.pct_improvement:.1f}%  "
          f"95% CI [{pct_lo:.1f}%, {pct_hi:.1f}%]  "
          f"{'(paired block bootstrap: CI excludes 0)' if significant else '(n.s. — paired CI includes 0)'}")
    print(f"  │  Per frame   : {results.rmse_before_vessel_perframe*100:.3f} "
          f"→ {results.rmse_after_vessel_perframe*100:.3f} cm/s  "
          f"({results.pct_improvement_perframe:+.1f}%)")
    print(f"  │  NOISE-FREE  : {results.rmse_systematic_before_vessel*100:.4f} "
          f"→ {results.rmse_systematic_after_vessel*100:.3e} cm/s  "
          f"({results.pct_improvement_systematic:+.4f}%)  ← quote this one")
    print(f"  │  MAE before  : {results.mae_before_vessel*100:.3f} cm/s")
    print(f"  │  MAE after   : {results.mae_after_vessel*100:.3f} cm/s")
    # Pearson: report PER COMPONENT, and report the whole-sphere version as
    # the primary one. On the vessel mask the true velocity is a near-uniform
    # plateau (the mask keeps only the core voxels), so its spatial variance
    # is ~0 and the correlation there is undefined -> NaN. The whole-sphere
    # set spans the full range of Lambda and is always well conditioned.
    # Pearson r is NOT reported as a result. r is invariant under linear
    # scaling, and gradient nonlinearity is a MULTIPLICATIVE distortion
    # (v_meas = Lambda(r).v_true), so wherever Lambda is roughly constant
    # across the ROI r sits near 1 both before and after correction by
    # construction and carries no information (vessel mask: 0.9996 ->
    # 0.9998). Worse, r stays blind to the systematic bias the correction
    # removes while remaining sensitive to the noise amplification that
    # Lambda^-1 introduces where kappa(Lambda) is large, so components can
    # DECREASE after a correction that demonstrably works. Reporting it would
    # invite a defence of a metric that cannot measure the effect under
    # study. RMSE / MAE / p95 above are error metrics and are the ones to
    # quote. Values are still computed and still written to run_config.json
    # as a reproducibility record.
    print("  │  Pearson r: not reported: r is scale-invariant, so it cannot "
          "measure a")
    print("  │    multiplicative Lambda distortion. Use RMSE / MAE / p95 "
          "above.")
    print("  ├─ diagnostic (whole sphere r≤R0, incl. empty-voxel noise amplification) ┤")
    print(f"  │  RMSE before : {results.rmse_before_whole*100:.3f} cm/s  "
          f"95% CI [{bw_lo*100:.3f}, {bw_hi*100:.3f}]")
    print(f"  │  RMSE after  : {results.rmse_after_whole*100:.3f} cm/s  "
          f"95% CI [{aw_lo*100:.3f}, {aw_hi*100:.3f}]")
    print(f"  │  Improvement : {results.pct_improvement_whole:.1f}%  "
          "(negative = correction amplifies edge noise in empty space)")
    if mode == 1:
        print(f"  │  Flow rate ref  : {results.flow_rate_ref:.1f} ml/min")
        print(f"  │  Flow rate std  : {results.flow_rate_std:.1f} ml/min")
        print(f"  │  Flow rate cor  : {results.flow_rate_cor:.1f} ml/min")
    print("  └───────────────────────────────────────────────────────┘")

    # ── Radial stratification ─────────────────────────────────────────────
    # Filter to vessel voxels (|v_true| > VESSEL_THRESH) so the per-bin RMSE
    # answers "how well does the correction recover real flow?" rather
    # than "how well does it preserve noise in empty space?".
    vt_mag      = np.linalg.norm(vt_avg, axis=0)        # (Nx,Ny,Nz)
    vessel_mask = vt_mag > VESSEL_THRESH               # adaptive (≤5 cm/s)
    n_vessel    = int(vessel_mask.sum())
    print(f"  {_fig_label(8)} Radial stratification: {n_vessel:,} vessel voxels "
          f"(|v_true| > {VTHR_LABEL})")
    r_mm, rmse_s, cnt = val_engine.radial_stratification(
        vm_avg, vt_avg, X, Y, Z, vessel_mask=vessel_mask)
    _,    rmse_c, _   = val_engine.radial_stratification(
        vc_avg, vt_avg, X, Y, Z, vessel_mask=vessel_mask)

    # ── Bland–Altman ──────────────────────────────────────────────────────
    bias_s, lo_s, hi_s, ms, ds = val_engine.bland_altman(vm_avg, vt_avg, mask=inside_r0)
    bias_c, lo_c, hi_c, mc, dc = val_engine.bland_altman(vc_avg, vt_avg, mask=inside_r0)

    # ── Peak-flow voxel for temporal profile plot ─────────────────────────
    # Pick the voxel with maximum mean velocity magnitude in v_true,
    # restricted to voxels WELL INSIDE the R0 calibration sphere
    # (specifically r ≤ 0.7·R0).
    #
    # Why 0.7·R0 and not just R0:
    # Even just inside R0 (at r/R0 ≈ 0.95), the Λ matrix is still
    # ill-conditioned (κ can reach 5–15), and the small vy/vz swirl
    # components (~7 cm/s) are below the amplified noise floor.
    # At r/R0 ≤ 0.7 the condition number is typically κ < 2, so the
    # correction works essentially perfectly there.
    # The vessel runs along x at y=z=0 so it passes through this safe
    # zone — we are guaranteed to find a vessel voxel inside the
    # tighter sphere.
    vmag_mean  = np.linalg.norm(v_true, axis=0).mean(axis=-1)  # (Nx,Ny,Nz)
    r_voxel    = np.sqrt(X**2 + Y**2 + Z**2)                     # (Nx,Ny,Nz)
    safe_zone  = r_voxel <= 0.7 * grad_coeffs.R0
    if safe_zone.any():
        masked   = np.where(safe_zone, vmag_mean, -np.inf)
        peak_idx = np.unravel_index(np.argmax(masked), masked.shape)
        sel_note = f"r ≤ 0.7·R₀ = {0.7*grad_coeffs.R0*1000:.0f} mm"
    elif inside_r0 is not None and inside_r0.any():
        masked   = np.where(inside_r0, vmag_mean, -np.inf)
        peak_idx = np.unravel_index(np.argmax(masked), masked.shape)
        sel_note = "inside R₀"
    else:
        peak_idx = np.unravel_index(vmag_mean.argmax(), vmag_mean.shape)
        sel_note = "no R₀ restriction"
    ix   = int(peak_idx[0])
    iy_p = int(peak_idx[1])
    iz_  = int(peak_idx[2])
    r_peak = float(r_voxel[ix, iy_p, iz_]) * 1000  # mm
    print(f"  [fig05] Peak-flow voxel ({sel_note}): "
          f"ix={ix}, iy={iy_p}, iz={iz_}")
    print(f"          physical: x={X[ix,iy_p,iz_]*1000:+.1f} mm  "
          f"y={Y[ix,iy_p,iz_]*1000:+.1f} mm  z={Z[ix,iy_p,iz_]*1000:+.1f} mm  "
          f"(r={r_peak:.1f} mm)")
    print(f"          |v|_mean = {vmag_mean[ix,iy_p,iz_]*100:.1f} cm/s")

    # ── Generate all figures ──────────────────────────────────────────────
    print("\n  [figures] Generating publication-quality figures …")

    plotter.plot_pct_error(v_true, v_meas, v_cor,
                            xv_mm, iy, t_sys, label,
                            inside_r0=inside_r0,
                            vessel_threshold_m_s=VESSEL_THRESH,
                            snr_db=(float(cfg.snr_db)
                                    if cfg.add_noise else None),
                            steady=(mode == 1 and not getattr(
                                cfg, "mode1_pulsatile", False)))
    plotter.plot_summary_metrics(results, label)          # Fig 12 — before 13/14

    return results



# ═════════════════════════════════════════════════════════════════════════════
# SNR SENSITIVITY SWEEP
# ═════════════════════════════════════════════════════════════════════════════

def run_snr_sweep(
    X: np.ndarray, Y: np.ndarray, Z: np.ndarray,
    Lambda: np.ndarray,
    cfg: SimulationConfig,
    corrector: "VelocityCorrector",
    val_engine: "ValidationEngine",
    inside_r0: Optional[np.ndarray] = None,
    snr_list: Tuple[float, ...] = (20.0, 25.0, 30.0, 40.0, 50.0),
    mode: int = 2,
) -> List[dict]:
    """
    Repeat the Mode-2 experiment across a range of SNR values.

    WHY THIS EXISTS. The headline percentage improvement of a synthetic GNL
    experiment is NOT a property of the correction. After correction the
    residual is essentially pure noise, so the "after" term is set by the
    assumed SNR while the "before" term is set by the (fixed) systematic
    distortion. The same code therefore reports a much lower improvement at
    20 dB than at 50 dB with nothing about the method having changed.

    A single number is consequently indefensible on its own. This sweep is
    what should appear in the thesis next to it, together with the noise-free
    row from run_simulation_mode(), which is the SNR-independent result.

    Choosing the operating point: measure sigma_v in a static region of the
    acquired data and invert the noise model,

        SNR_lin = sqrt(2) * VENC / (pi * sigma_v),   SNR_dB = 20*log10(SNR_lin)

    e.g. sigma_v = 3.99 cm/s at VENC = 90 cm/s gives SNR_lin ~ 10.2, i.e.
    ~20 dB — which is the regime real 4D flow actually operates in, and far
    from the 50 dB default.

    Returns a list of per-SNR dicts (also printed as a table).
    """
    sim = SimulationEngine4D(cfg)
    if mode == 1:
        v_true = sim.generate_poiseuille_4d(X, Y, Z)
    else:
        v_true = sim.generate_pulsatile_vessel_4d(X, Y, Z)

    v_pk  = float(np.linalg.norm(v_true, axis=0).max())
    thr   = max(PCT_ERROR_VESSEL_FRACTION * v_pk, 1e-4)
    vt_a  = v_true.mean(axis=-1)
    vessel = np.linalg.norm(vt_a, axis=0) > thr
    if inside_r0 is not None:
        vessel = vessel & inside_r0

    v_meas0 = SimulationEngine4D.apply_forward_distortion(v_true, Lambda)

    print("\n" + "─" * 74)
    print("  [SNR sweep] The percentage improvement is a NOISE statistic. "
          "Same correction,")
    print("              same Λ, only the assumed SNR changes:")
    print(f"              vessel mask n = {int(vessel.sum()):,}  "
          f"(|v_true| > {thr*100:.2f} cm/s, time average)")
    print("     SNR    sigma_v     RMSE before → after   time-avg    "
          "per-frame")
    print("     [dB]   [cm/s]            [cm/s]           impr.        impr.")

    rows = []
    snr_saved = cfg.snr_db
    try:
        for snr in snr_list:
            cfg.snr_db = float(snr)
            sigma = np.sqrt(2.0) * cfg.VENC / (np.pi * 10 ** (snr / 20.0))
            # Fixed seed across the sweep so the only variable is the noise
            # AMPLITUDE, not the particular realisation.
            rng = np.random.default_rng(seed=20250 + mode)
            v_m = v_meas0 + rng.normal(0.0, sigma, v_meas0.shape)
            v_c = corrector.correct_4d(v_m, Lambda)

            rb_t = val_engine.rmse(v_m.mean(axis=-1), vt_a, mask=vessel)
            ra_t = val_engine.rmse(v_c.mean(axis=-1), vt_a, mask=vessel)
            rb_f = float(np.sqrt(np.mean(
                np.sum((v_m - v_true) ** 2, axis=0)[vessel])))
            ra_f = float(np.sqrt(np.mean(
                np.sum((v_c - v_true) ** 2, axis=0)[vessel])))
            row = {
                "snr_db": float(snr),
                "sigma_v_cm_s": float(sigma * 100),
                "rmse_before_cm_s": float(rb_t * 100),
                "rmse_after_cm_s":  float(ra_t * 100),
                "pct_improvement_time_avg": val_engine.pct_improvement(rb_t, ra_t),
                "rmse_before_perframe_cm_s": rb_f * 100,
                "rmse_after_perframe_cm_s":  ra_f * 100,
                "pct_improvement_perframe": val_engine.pct_improvement(rb_f, ra_f),
            }
            rows.append(row)
            print(f"     {snr:4.0f}   {sigma*100:7.3f}   "
                  f"{rb_t*100:7.3f} → {ra_t*100:7.3f}   "
                  f"{row['pct_improvement_time_avg']:+7.1f}%   "
                  f"{row['pct_improvement_perframe']:+7.1f}%")
    finally:
        cfg.snr_db = snr_saved

    print("              A negative per-frame value means Λ⁻ᵀ amplifies the "
          "noise faster than it")
    print("              removes the distortion — which is the honest answer "
          "at that SNR for a")
    print("              vessel this close to isocentre, and is exactly why "
          "the noise-free row")
    print("              is the one to quote for the CORRECTION.")
    print("─" * 74 + "\n")
    return rows


# ═════════════════════════════════════════════════════════════════════════════
# SELF-TESTS
# ═════════════════════════════════════════════════════════════════════════════

def run_self_tests(cfg: SimulationConfig,
                   grad_coeffs: Optional["GradientCoefficients"] = None) -> bool:
    """
    Minimal correctness suite for the core simulation pipeline.

    `grad_coeffs`, when given, is the ACTUAL coil file loaded for this run.
    T5 then perturbs those coefficients instead of a four-term toy coil, so
    the robustness number that gets quoted in the thesis is the robustness of
    the real gradient system, not of a stand-in. Results are recorded in the
    module-level SELF_TEST_RESULTS dict and written to run_config.json.

    Tests
    -----
    T1  — Identity Lambda: no distortion → corrected == measured == true.
    T2  — Round-trip: correct_4d(apply_forward(v, Λ), Λ) ≈ v_true, with a
          non-symmetric Λ so that a Λ/Λᵀ mix-up would fail.
    T3  — Spherical-harmonic evaluator vs the closed-form derivative of a
          known A(2,0) harmonic (tests Schmidt norm + Legendre + stencil).
    T3b — Regression: the primary linear term A(1,1)/B(1,1)/A(1,0) must not
          be counted twice, i.e. Λ(0,0,0) = I for a file that lists it.
    T4  — Poiseuille peak velocity matches the configured value on the grid.
    T5  — Model mismatch: distort with one coefficient set and correct with
          sets perturbed by 2, 5 and 10 % (per-coefficient Gaussian, one
          draw per level, random velocities on a cube of half-width
          R0/sqrt(3)). T1/T2 are circular by construction; T5 is not,
          and is the test to quote when defending the validation.

    Returns True if all tests pass, False otherwise (prints diagnostics).
    """
    print("\n" + "─" * 60)
    print("  [self-test] Running pipeline correctness checks …")
    passed = []

    # Self-tests verify the EXACT inverse algorithm at machine precision.
    # Tikhonov regularisation deliberately introduces a small bias to reduce
    # noise — that bias would fail T1 / T2 by construction. We therefore
    # build the test corrector with λ = 0 (exact inverse), while leaving
    # the production cfg.tikhonov_lambda value untouched.
    test_cfg = copy.copy(cfg)
    test_cfg.tikhonov_lambda = 0.0
    corrector  = VelocityCorrector(test_cfg)
    sim        = SimulationEngine4D(cfg)
    # Seeded generator: an unseeded np.random made the pass/fail outcome
    # depend on global RNG state, so a failure could not be reproduced.
    rng        = np.random.default_rng(20240517)
    tol_tight  = 1e-6    # m/s — numerical tolerance for exact tests
    tol_loose  = 1e-4    # m/s — tolerance for noise-free distortion round-trip

    # ── T1: Identity Lambda ───────────────────────────────────────────────
    try:
        N = 8
        sp = (N, N, N)
        v_rand = rng.normal(0, 0.1, (3,) + sp + (3,))
        # Identity distortion matrix
        L_id = np.zeros((3, 3) + sp)
        for i in range(3):
            L_id[i, i] = 1.0
        v_fwd = SimulationEngine4D.apply_forward_distortion(v_rand, L_id)
        v_cor = corrector.correct_4d(v_fwd, L_id)
        err   = float(np.abs(v_cor - v_rand).max())
        ok    = err < tol_tight
        passed.append(ok)
        print(f"  [T1] Identity-Lambda round-trip:  max_err = {err:.2e} m/s  "
              f"{'PASS ✓' if ok else 'FAIL ✗'}")
    except Exception as e:
        passed.append(False)
        print(f"  [T1] EXCEPTION: {e}")

    # ── T2: Real 3×3 Lambda round-trip (noise-free) ───────────────────────
    try:
        N = 10
        sp = (N, N, N)
        # Construct a mild but non-trivial distortion matrix (≈5% off-diagonal)
        L_test = np.zeros((3, 3) + sp)
        for i in range(3):
            L_test[i, i] = 1.0 + 0.03 * rng.uniform(-1, 1, sp)
        for i in range(3):
            for j in range(3):
                if i != j:
                    L_test[i, j] = 0.05 * rng.uniform(-1, 1, sp)

        v_true = rng.normal(0, 0.5, (3,) + sp + (5,))
        v_fwd  = SimulationEngine4D.apply_forward_distortion(v_true, L_test)
        v_cor  = corrector.correct_4d(v_fwd, L_test)
        err    = float(np.abs(v_cor - v_true).max())
        ok     = err < tol_loose
        passed.append(ok)
        print(f"  [T2] Non-trivial-Lambda round-trip:  max_err = {err:.2e} m/s  "
              f"{'PASS ✓' if ok else 'FAIL ✗'}")
    except Exception as e:
        passed.append(False)
        print(f"  [T2] EXCEPTION: {e}")

    # ── T3: spherical-harmonic evaluator vs CLOSED-FORM derivative ───────
    # An earlier T3 built a coefficient set with NO harmonics at all, so the
    # SH loop never executed and the test could not detect an error in the
    # harmonic machinery. It now checks Λ against an analytically
    # differentiable harmonic, which exercises the Schmidt normalisation,
    # the Legendre evaluation and the central-difference stencil together.
    #
    # For the z-coil with a single A(2,0) coefficient c:
    #   N_20 = 1,  P_2^0(cosθ) = (3cos²θ − 1)/2
    #   Bz   = z/R0 + c·(r/R0)²·(3cos²θ − 1)/2
    #        = z/R0 + c·(2z² − x² − y²)/(2R0²)
    # so, multiplying by R0 as compute_lambda does,
    #   Λ_zz = 1 + 2cz/R0,   Λ_xz = −cx/R0,   Λ_yz = −cy/R0.
    try:
        c_test = 0.02
        synthetic = GradientCoefficients(
            filepath="<synthetic A(2,0) harmonic>",
            R0=0.25,
            coeffs={"x": [], "y": [], "z": [("A", 2, 0, c_test)]},
            coil_name="synthetic_A20",
            max_degree=2,
            lnorm=4,
        )
        ev  = GradientFieldEvaluator(synthetic)
        R0t = synthetic.R0
        pts = np.array([[0.05, -0.03, 0.08],
                        [0.10,  0.06, -0.12],
                        [-0.07, 0.09,  0.02]])
        Xp = pts[:, 0].reshape(-1, 1, 1)
        Yp = pts[:, 1].reshape(-1, 1, 1)
        Zp = pts[:, 2].reshape(-1, 1, 1)
        Lp = ev.compute_lambda(Xp, Yp, Zp, eps=5e-4)
        err = 0.0
        for k in range(pts.shape[0]):
            xk, yk, zk = pts[k]
            err = max(err, abs(Lp[2, 2, k, 0, 0] - (1.0 + 2 * c_test * zk / R0t)))
            err = max(err, abs(Lp[0, 2, k, 0, 0] - (-c_test * xk / R0t)))
            err = max(err, abs(Lp[1, 2, k, 0, 0] - (-c_test * yk / R0t)))
        ok = err < 1e-6
        passed.append(ok)
        print(f"  [T3] SH evaluator vs closed-form dBz/dj:  max_err = {err:.2e}  "
              f"{'PASS ✓' if ok else 'FAIL ✗'}")
    except Exception as e:
        passed.append(False)
        print(f"  [T3] EXCEPTION: {e}")

    # ── T3b: the primary linear term must not be counted twice ───────────
    # Siemens .grad files list the primary term (A(1,1)/B(1,1)/A(1,0)) with
    # value 1.0. _bz_field also adds the ideal linear term analytically, so
    # if the parser kept the listed coefficient the two would sum and give
    # Λ_ii = 2 — halving every corrected velocity, silently. This is a
    # regression test for GradientFieldParser._PRIMARY_TERM.
    try:
        raw = ("CoilName = regression_test\n"
               "0.25 m = R0\n"
               "lnorm = 4\n"
               "  1 A( 1, 1)   1.0000000 x\n"
               "  2 B( 1, 1)   1.0000000 y\n"
               "  3 A( 1, 0)   1.0000000 z\n")
        tmp = Path(os.environ.get("TEMP", "/tmp")) / "_gnl_primary_test.grad"
        tmp.write_text(raw, encoding="utf-8")
        gc_t = GradientFieldParser.parse(str(tmp))
        ev_t = GradientFieldEvaluator(gc_t)
        O    = np.array([[[0.0]]])
        L0   = ev_t.compute_lambda(O, O, O, eps=1e-4)[:, :, 0, 0, 0]
        err  = float(np.abs(L0 - np.eye(3)).max())
        ok   = err < 1e-6
        passed.append(ok)
        print("  [T3b] Primary term not double-counted, Λ(0,0,0) = I:  "
              f"max_err = {err:.2e}  {'PASS ✓' if ok else 'FAIL ✗'}")
        try:
            tmp.unlink()
        except OSError:
            pass
    except Exception as e:
        passed.append(False)
        print(f"  [T3b] EXCEPTION: {e}")

    # ── T4: Poiseuille peak velocity analytical check ─────────────────────
    # The discrete maximum of a parabolic profile sampled on a Cartesian
    # grid systematically under-estimates v_max because no voxel centre
    # lies exactly on the tube axis. This is the same partial-volume effect
    # Watanabe et al. (2019) measured on real scanners: they report
    # under-estimation of the peak velocity and over-estimation near the
    # tube wall on all three vendors, citing the criterion that the peak
    # velocity error stays below 10 % when the voxel-to-inner-diameter
    # ratio is below 10 % (criterion originally from Fukuyama et al. 2017,
    # their ref 16).
    #
    # Check that ratio for the configured grid — at the default 97³ over
    # ±220 mm the voxel is ~4.6 mm against a 60 mm tube diameter, i.e. ~8 %,
    # inside Watanabe's threshold. Coarser grids (e.g. 48³: ~15 %) exceed it,
    # which is why a few percent under-shoot is tolerated there rather than
    # treated as a failure; the tolerance widens for coarser grids.
    try:
        r = cfg.tube_radius_m
        # Peak velocity is now set directly by config (see generate_poiseuille_4d);
        # T4 checks that the discrete sampled peak matches the configured peak.
        v_max_analytical = cfg.poiseuille_v_peak_m_s

        N  = cfg.grid_N
        ax = np.linspace(-cfg.r_max_m, cfg.r_max_m, N)
        X_, Y_, Z_ = np.meshgrid(ax, ax, ax, indexing="ij")
        v_pois = sim.generate_poiseuille_4d(X_, Y_, Z_)

        # Peak x-velocity across all voxels and times
        v_max_sim = float(v_pois[0].max())
        rel_err   = abs(v_max_sim - v_max_analytical) / max(v_max_analytical, 1e-12)

        # Grid-aware tolerance: 5 % at production density (6+ voxels across
        # diameter), gracefully widened for coarse grids.
        # Spacing of linspace(-r_max, r_max, N) is 2*r_max/(N-1), not /N.
        voxel_size_mm  = float(abs(ax[1] - ax[0])) * 1000.0
        vox_across_dia = 2 * r * 1000 / voxel_size_mm
        # Tolerance: 5 % at 6+ voxels across, +5 % per voxel deficit below 6
        tol = 0.05 + max(0.0, 6.0 - vox_across_dia) * 0.05
        ok  = rel_err < tol
        passed.append(ok)
        warning = ""
        if vox_across_dia < 5.0 and ok:
            warning = ("  (grid under-resolves tube — only "
                       f"{vox_across_dia:.1f} voxels across diameter; "
                       f"tolerance widened to {tol*100:.0f} %)")
        print(f"  [T4] Poiseuille v_max:  analytical = {v_max_analytical*100:.3f} cm/s  "
              f"sim_max = {v_max_sim*100:.3f} cm/s  "
              f"rel_err = {rel_err*100:.2f}%  "
              f"{'PASS ✓' if ok else 'FAIL ✗'}{warning}")
    except Exception as e:
        passed.append(False)
        print(f"  [T4] EXCEPTION: {e}")

    # ── T5: model-mismatch robustness (non-circular validation) ──────────
    # WHY THIS TEST EXISTS.
    # T1/T2 invert exactly the operator they applied, so they can only
    # confirm the linear algebra — they cannot show that the correction
    # helps when the model is imperfect. Every simulation-only validation of
    # a GNL correction shares this weakness: the forward model and the
    # correction use the same Λ, so a perfect result is guaranteed by
    # construction and says nothing about real data.
    #
    # T5 breaks that loop. The field is distorted with one coefficient set
    # ("the scanner") and corrected with a PERTURBED set ("the calibration
    # file we were given"), mimicking the realistic case where the vendor
    # coefficients do not describe the coil exactly. A correction that only
    # works when the model is exact would show no benefit here.
    #
    # Report this test in the thesis alongside the headline RMSE: it is the
    # evidence that the improvement is not an artefact of self-inversion.
    #
    # WHICH COIL T5 USES. If the real coefficient file is available it is
    # perturbed directly, so the quoted robustness is the robustness of THIS
    # gradient system. A four-term toy coil (the fallback below) exercises the
    # same code path but its answer is not transferable to the thesis.
    #
    # The perturbation is a per-coefficient Gaussian of the stated relative
    # size, not a uniform rescale: a uniform scale factor is a degenerate
    # error mode that partly cancels in Lambda, so it makes the test easier
    # than reality. Several magnitudes are swept because "does it still help"
    # is less informative than "how fast does it degrade".
    try:
        _t5_rows = []
        if grad_coeffs is not None and any(grad_coeffs.coeffs.values()):
            R0t   = grad_coeffs.R0
            _src  = f"real coil file ({grad_coeffs.coil_name})"

            def _coil(pert, seed):
                """Real coefficients with a per-term Gaussian relative error."""
                _r = np.random.default_rng(seed)
                return GradientCoefficients(
                    filepath="<T5-perturbed>", R0=R0t,
                    coeffs={ax: [(t, n, m, v * (1.0 + pert * _r.standard_normal()))
                                 for (t, n, m, v) in grad_coeffs.coeffs[ax]]
                            for ax in ("x", "y", "z")},
                    coil_name=f"{grad_coeffs.coil_name}+{pert:.0%}",
                    max_degree=grad_coeffs.max_degree, lnorm=4)
        else:
            R0t   = 0.25
            _src  = "synthetic 4-term toy coil (no real file supplied)"
            base  = [("A", 2, 0,  0.030), ("A", 3, 1, -0.018),
                     ("B", 2, 2,  0.012), ("A", 4, 0,  0.009)]

            def _coil(pert, seed):
                _r = np.random.default_rng(seed)
                _s = lambda v: v * (1.0 + pert * _r.standard_normal())
                return GradientCoefficients(
                    filepath="<T5>", R0=R0t,
                    coeffs={"x": [(t, n, m,  _s(v)) for t, n, m, v in base],
                            "y": [(t, n, m, -_s(v)) for t, n, m, v in base],
                            "z": [(t, n, m,  _s(v)) for t, n, m, v in base]},
                    coil_name="T5", max_degree=4, lnorm=4)

        # Sample the region the expansion is valid in, on a grid coarse enough
        # to stay fast: r <= R0/sqrt(3) puts every corner inside the sphere.
        n_ax = 12
        _half = R0t / np.sqrt(3.0)
        axl  = np.linspace(-_half, _half, n_ax)
        Xt, Yt, Zt = np.meshgrid(axl, axl, axl, indexing="ij")

        L_true = GradientFieldEvaluator(_coil(0.0, 0)).compute_lambda(
            Xt, Yt, Zt, eps=5e-4)
        v_t = rng.normal(0.0, 0.30, (3,) + Xt.shape + (2,))
        v_m = SimulationEngine4D.apply_forward_distortion(v_t, L_true)
        rmse_before = float(np.sqrt(np.mean((v_m - v_t) ** 2)))

        print(f"  [T5] Model mismatch — coefficients from the {_src}")
        print("       perturbation   RMSE before → after [cm/s]   improvement")
        ok_all = True
        for _p, _seed in ((0.02, 101), (0.05, 102), (0.10, 103)):
            L_model = GradientFieldEvaluator(_coil(_p, _seed)).compute_lambda(
                Xt, Yt, Zt, eps=5e-4)
            v_c = corrector.correct_4d(v_m, L_model)   # corrected with WRONG Λ
            rmse_after = float(np.sqrt(np.mean((v_c - v_t) ** 2)))
            improve = 100.0 * (rmse_before - rmse_after) / max(rmse_before, 1e-12)
            _ok = rmse_after < rmse_before
            ok_all &= _ok
            _t5_rows.append({"coeff_error_pct": _p * 100.0,
                             "rmse_before_cm_s": rmse_before * 100.0,
                             "rmse_after_cm_s":  rmse_after * 100.0,
                             "improvement_pct":  improve,
                             "pass": bool(_ok)})
            print(f"       {_p:>10.0%}   {rmse_before*100:8.3f} → "
                  f"{rmse_after*100:8.3f}        {improve:+6.1f}%   "
                  f"{'PASS ✓' if _ok else 'FAIL ✗'}")
        passed.append(ok_all)
        print("       The correction still helps with an imperfect "
              "calibration file — this is the")
        print("       non-circular evidence, unlike T1/T2. Quote this table "
              "in the validation chapter.")
        SELF_TEST_RESULTS["T5_model_mismatch"] = {
            "coefficient_source": _src,
            "grid_half_extent_m": float(_half),
            "rows": _t5_rows,
        }
    except Exception as e:
        passed.append(False)
        print(f"  [T5] EXCEPTION: {e}")
        SELF_TEST_RESULTS["T5_model_mismatch"] = {"exception": str(e)}

    n_pass = sum(passed)
    n_total = len(passed)
    print(f"  [self-test] {n_pass}/{n_total} tests passed"
          + ("  — ALL OK" if n_pass == n_total else "  — REVIEW FAILURES ABOVE"))
    print("─" * 60 + "\n")
    SELF_TEST_RESULTS["n_passed"] = int(n_pass)
    SELF_TEST_RESULTS["n_total"]  = int(n_total)
    SELF_TEST_RESULTS["all_passed"] = bool(n_pass == n_total)
    return n_pass == n_total


# ═════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════

def _build_brain_mask(mag_vol, frac=0.10, erode=4):
    """
    Crude but effective anatomical brain mask from a magnitude volume.

    Scalp fat, orbital tissue and skull-base structures are BRIGHT on magnitude
    but carry corrupted phase (fat shift, motion, pulsation ghosting), so they
    pass any velocity-SNR threshold: they are bright AND fast, exactly like a
    vessel. They can only be removed anatomically.

    On MR the cortical skull is signal-void, so scalp and brain are separated
    by a dark ring. Eroding the magnitude mask therefore breaks the thin scalp
    shell apart while the solid brain survives; the largest remaining connected
    component is the brain. Dilating back restores the original extent.

    Returns a boolean array, or None if scipy is unavailable or the result
    looks implausible, in which case the caller falls back to the plain
    magnitude mask.
    """
    try:
        from scipy import ndimage as ndi
    except Exception:
        print("  [DICOM] scipy.ndimage unavailable - brain mask skipped.")
        return None
    try:
        m = mag_vol > (frac * float(np.nanmax(mag_vol)))
        if m.sum() < 1000:
            return None
        er = max(1, int(erode))
        m_er = ndi.binary_erosion(m, iterations=er)
        lab, n = ndi.label(m_er)
        if n == 0:
            print("  [DICOM] Brain mask: erosion removed everything - skipped.")
            return None
        sizes = ndi.sum(m_er, lab, index=range(1, n + 1))
        brain = (lab == (int(np.argmax(sizes)) + 1))
        brain = ndi.binary_dilation(brain, iterations=er)
        brain = ndi.binary_fill_holes(brain)
        frac_kept = float(brain.sum()) / float(max(m.sum(), 1))
        # Sanity: the brain should be a large majority of the head signal but
        # not all of it. Outside that range the component search has latched
        # onto the wrong structure and the mask is worse than none.
        if not (0.15 < frac_kept < 0.98):
            print(f"  [DICOM] Brain mask rejected: kept {100*frac_kept:.0f}% of "
                  "signal voxels, outside the plausible 15-98% range. "
                  "Using the magnitude mask instead.")
            return None
        print(f"  [DICOM] Brain mask: {int(brain.sum()):,} voxels "
              f"({100*frac_kept:.0f}% of magnitude-signal voxels, erode={er}). "
              "Scalp/orbit/skull-base excluded.")
        return brain
    except Exception as exc:
        print(f"  [DICOM] Brain mask failed ({exc}) - continuing without it.")
        return None


def _dicom_quick_plot(
    v_meas: np.ndarray,
    v_cor:  np.ndarray,
    mask:   Optional[np.ndarray],
    output_dir: str,
    coil:   str = "",
    lambda_dev: Optional[np.ndarray] = None,
    Xd: Optional[np.ndarray] = None,
    Yd: Optional[np.ndarray] = None,
    Zd: Optional[np.ndarray] = None,
    mag_vol: Optional[np.ndarray] = None,
    venc:    Optional[float] = None,
    vessel_thr: Optional[float] = None,
    vessel_sigma: Optional[float] = None,
    vessel_stats: Optional[dict] = None,
    flow_gate: Optional[np.ndarray] = None,
    fig05_slice_override: Optional[int] = None,
    exclude_outlier_planes: bool = True,
) -> None:
    """
    Comprehensive fig05 for the DICOM correction stage.

    Eight panels:
      (a) Histogram of |v| before vs after correction (inside r ≤ R₀)
      (b-d) Max-correction axial slice: |v| measured, |v| corrected, |Δv|
      (e) Mean correction magnitude per slice
      (g) Correction magnitude vs distance from isocenter (radial profile)
      (f) Summary text box with key statistics

    All comparisons are between v_meas and v_corrected — there is no
    ground truth for acquired data. The figure shows:
      · The DISTRIBUTION shift (how the correction changed velocities)
      · The SPATIAL pattern of corrections (largest at FOV periphery)
      · The PER-SLICE magnitude of corrections
    These are the correct metrics for acquired-data validation.
    """
    # ── Shape handling ────────────────────────────────────────────────────
    # v_meas / v_cor: (3, Nx, Ny, Nz, Nt)
    if v_meas.ndim == 4:                      # single-axis legacy shape
        v_meas = v_meas[np.newaxis]
        v_cor  = v_cor[np.newaxis]

    Nax, Nx, Ny, Nz, Nt = v_meas.shape
    t_sys  = max(0, Nt // 3)                  # ~systole frame
    iz_mid = Nz // 2                           # central axial slice

    # Velocity magnitudes — average over cardiac phases
    # Built one cardiac phase at a time, for the same reason as absd_4d
    # below: np.linalg.norm(x, axis=0) allocates a temporary the size of its
    # INPUT before reducing, so these two calls alone used to add 2 x 556 MiB
    # of transient allocation on a 3x224x224x88x11 volume — on top of v_meas
    # and v_cor, which are still live. That put fig05's peak at ~2.2 GiB on
    # the same machine where the statistics block died at 3.0 GiB. Streaming
    # keeps the transient to a single 3-D frame (~17 MiB).
    def _mag_4d_streamed(v5):
        out = np.empty(v5.shape[1:], dtype=np.float32)
        _a = np.empty(v5.shape[1:4], dtype=np.float32)
        _t = np.empty(v5.shape[1:4], dtype=np.float32)
        for _k in range(v5.shape[-1]):
            _a[:] = 0.0
            for _i in range(v5.shape[0]):
                np.square(v5[_i, ..., _k], out=_t, dtype=np.float32)
                _a += _t
            np.sqrt(_a, out=_a)
            out[..., _k] = _a
        return out

    mag_m_4d = _mag_4d_streamed(v_meas)                # (Nx,Ny,Nz,Nt)
    mag_c_4d = _mag_4d_streamed(v_cor)
    # ── correction magnitude: norm of the DIFFERENCE VECTOR ───────────────
    # NOT the difference of the norms. The two coincide only if the
    # correction is a pure rescaling along the original direction; Lambda^-1
    # also ROTATES the vector, so
    #        ||v_cor - v_meas||  >=  | |v_cor| - |v_meas| |
    # and the gap between them is exactly the direction error quantified by
    # direction_error_stats(). This figure previously used the difference of
    # norms while the stats block used the vector norm, so the text box and
    # the console log disagreed by a constant factor (~1.8x on the Circle of
    # Willis data, mean 0.076 vs 0.135 cm/s) even though both were labelled
    # |v_cor - v_meas|. The vector norm is the quantity the labels claim and
    # the one every quoted number uses, so the figure now matches.
    #
    # Built one cardiac phase at a time: v_cor - v_meas over all phases would
    # allocate a second full (3, Nx, Ny, Nz, Nt) array.
    absd_4d = np.empty_like(mag_m_4d)
    for _t in range(Nt):
        absd_4d[..., _t] = np.linalg.norm(v_cor[..., _t] - v_meas[..., _t],
                                          axis=0)

    mag_m_3d = mag_m_4d.mean(axis=-1)                   # (Nx,Ny,Nz)
    mag_c_3d = mag_c_4d.mean(axis=-1)
    absd_3d  = absd_4d.mean(axis=-1)

    # Build spatial mask broadcast
    if mask is None:
        mask = np.ones((Nx, Ny, Nz), dtype=bool)

    # ── Choose the display slice: the one containing the most flow ──────────
    # The |v| panels exist to show the actual flow field. GNL correction on this
    # data is a ~2% effect that is invisible at the 0-30 cm/s display scale on
    # ANY slice (it is shown instead in the stretched difference panel and the
    # radial profile). Peripheral slices maximise the correction but are exactly
    # where the magnitude signal fades and the phase degenerates into noise
    # (spurious >VENC velocities), producing meaningless speckle. So the slice
    # is chosen to show real flow, not maximum correction.
    #
    # Scoring by NUMBER OF MASKED VOXELS would pick the geometrically widest
    # slice. In a phantom that is also where the flow is, but in a head it is
    # mid-brain — while the Circle of Willis sits near the skull base. So the
    # slice is scored by FLOW CONTENT instead: the number of masked voxels
    # whose speed exceeds 30% of the volume's 99th-percentile speed. That
    # lands on the slice containing the arteries on both phantom and head.
    _n_sig = np.array([int(mask[:, :, k].sum()) for k in range(Nz)])
    _outlier_planes = np.zeros(Nz, dtype=bool)
    if exclude_outlier_planes and fig05_slice_override is None and Nz >= 8:
        # Robust (MAD-based) outlier test on each plane's in-plane bounding-box
        # extent (the larger of its two in-plane spans). This is far more
        # sensitive than a voxel-COUNT test: a corrupted/RF-contaminated plane
        # can add only a modest number of extra voxels (a thin bright streak
        # across the field) while still spanning several times the normal
        # cross-section - exactly the signature this catches; on a real 0.67 mm
        # pipe acquisition a genuine plane's extent varied by only +-1 voxel
        # (MAD 1.0) while the contaminated plane's extent was 4x the median.
        _ext = np.zeros(Nz)
        for _k in range(Nz):
            _jj, _kk = np.nonzero(mask[:, :, _k])
            if _jj.size:
                _ext[_k] = max(_jj.max() - _jj.min() + 1, _kk.max() - _kk.min() + 1)
        _present = _ext > 0
        if _present.sum() >= 8:
            _med = np.median(_ext[_present])
            _mad = np.median(np.abs(_ext[_present] - _med)) or 1.0
            _outlier_planes = _present & ((_ext - _med) > 4 * 1.4826 * _mad)
        if _outlier_planes.any():
            print(f"  [fig05] {int(_outlier_planes.sum())} plane(s) excluded from display-slice "
                  f"scoring as statistical outliers (indices {list(np.nonzero(_outlier_planes)[0])}): "
                  "masked-voxel count far above neighbouring planes, the signature of a corrupted "
                  "or RF-contaminated slice rather than real anatomy. Set exclude_outlier_planes="
                  "False to disable.")
    if mask.any() and np.isfinite(mag_m_3d[mask]).any():
        # 99.9th percentile, not 99th: the arteries occupy well under 1% of the
        # masked voxels in a head, so a lower percentile sits in stationary
        # tissue and drags the threshold down into noise. The floor at 10% of
        # VENC guards the same failure from the other side.
        _v_ref = float(np.percentile(mag_m_3d[mask], 99.9))
        # Threshold for "is this a flow voxel", used ONLY to score which slice
        # to display. Anchor it to the data's own velocity distribution, not to
        # VENC: when VENC is much larger than the flow (e.g. VENC=150 cm/s for
        # ~13 cm/s phantom flow) a VENC-relative floor lands above the flow and
        # scores every slice as empty. A small absolute floor (2 cm/s) only
        # keeps it out of pure noise.
        _thr   = max(0.40 * _v_ref, 2.0 / 100.0)
        _n_flow = np.array([
            int(((mag_m_3d[:, :, k] > _thr) & mask[:, :, k]).sum())
            for k in range(Nz)])
    else:
        _v_ref, _thr = 0.0, 0.0
        _n_flow = _n_sig
    if fig05_slice_override is not None:
        iz_show = int(fig05_slice_override)
        if not (0 <= iz_show < Nz):
            raise ValueError(f"--fig05-slice {iz_show} is out of range (this volume has {Nz} slices, "
                              f"valid indices 0..{Nz-1})")
        print(f"  [fig05] Display slice z={iz_show} — set manually (--fig05-slice): "
              f"{_n_flow[iz_show]:,} flow voxels above {_thr*100:.1f} cm/s, "
              f"{_n_sig[iz_show]:,} signal voxels")
    else:
        _cand_flow = np.where(_outlier_planes, -1, _n_flow)
        _cand_sig = np.where(_outlier_planes, -1, _n_sig)
        if _cand_flow.max() > 0:
            iz_show = int(_cand_flow.argmax())
        elif _cand_sig.max() > 0:
            iz_show = int(_cand_sig.argmax())
        else:
            iz_show = iz_mid
        print(f"  [fig05] Display slice z={iz_show} "
                  f"({_n_flow[iz_show]:,} flow voxels above "
                  f"{_thr*100:.1f} cm/s, {_n_sig[iz_show]:,} signal voxels "
                  "— automatic, by flow content)")

    # Flatten masked voxels (time-averaged)
    m_flat = mag_m_3d[mask]
    c_flat = mag_c_3d[mask]
    d_flat = absd_3d[mask]

    # ── Per-slice correction profile ──────────────────────────────────────
    # Restrict to the signal mask. Averaging over every voxel in the slice lets
    # the background dominate: it is the majority of the volume, its |v| is
    # noise spanning up to VENC, and |Δv| ≈ ‖Λ⁻ᵀ−I‖·|v| scales with that. The
    # unmasked curve therefore reported values an order of magnitude above the
    # figures in the stats box, which describe masked voxels only.
    if mask is not None and mask.any():
        _md = np.where(mask, absd_3d, np.nan)
        with np.errstate(invalid="ignore"):
            per_slice_delta = np.nanmean(_md, axis=(0, 1)) * 100
        per_slice_delta = np.nan_to_num(per_slice_delta, nan=0.0)
        _ps_basis = "signal voxels"
    else:
        per_slice_delta = absd_3d.mean(axis=(0, 1)) * 100
        _ps_basis = "all voxels"

    # ── Quantitative validation vs Λ-deviation map (saved as fig01) ──
    # The correction magnitude is |Δv| ≈ ‖(Λ⁻ᵀ−I)·v‖, i.e. it carries BOTH
    # the field deviation AND the local speed. Correlating the raw |Δv| map
    # against ‖Λ−I‖_F therefore understates the geometric agreement badly
    # wherever |v| varies more than position does (in a phantom the speed
    # spans two orders of magnitude while ‖Λ−I‖ changes only a few-fold):
    # the same speed confound this file already documents for the
    # radius correlation in main(). The PRIMARY metric is therefore the
    # RELATIVE correction |Δv|/|v| vs ‖Λ−I‖, which removes the speed factor
    # and isolates exactly the part the gradient model predicts. The absolute
    # correlation is kept as a secondary number for continuity with earlier
    # reports. Computed over the calibration-sphere/signal voxels; the
    # relative form additionally floors |v| at 0.5 cm/s so near-zero speeds
    # cannot blow the ratio up.
    lambda_corr_r   = None      # absolute |Δv|      vs ‖Λ−I‖ (speed-confounded)
    lambda_corr_rel = None      # relative |Δv|/|v|  vs ‖Λ−I‖ (primary)
    if lambda_dev is not None and lambda_dev.shape == absd_3d.shape:
        a = absd_3d[mask].ravel()
        b = lambda_dev[mask].ravel()
        if a.size > 10 and a.std() > 1e-12 and b.std() > 1e-12:
            lambda_corr_r = float(np.corrcoef(a, b)[0, 1])
        _spd  = mag_m_3d[mask].ravel()
        _flr  = 0.005                        # 0.5 cm/s divide-by-small floor
        _okr  = (np.isfinite(a) & np.isfinite(b) & np.isfinite(_spd)
                 & (_spd > _flr))
        if int(_okr.sum()) > 10:
            _rel = a[_okr] / _spd[_okr]
            if _rel.std() > 1e-12 and b[_okr].std() > 1e-12:
                lambda_corr_rel = float(np.corrcoef(_rel, b[_okr])[0, 1])

    # ── Key stats ─────────────────────────────────────────────────────────
    vmeas_mean  = float(m_flat.mean()           * 100)
    vcorr_mean  = float(c_flat.mean()           * 100)
    delta_mean  = float(d_flat.mean()           * 100)
    delta_med   = float(np.median(d_flat)       * 100)
    delta_p95   = float(np.percentile(d_flat,95)* 100)
    pct_change  = float(100.0 * delta_mean / max(vmeas_mean, 1e-6))

    # ── Figure layout ─────────────────────────────────────────────────────
    fig = plt.figure(figsize=(22, 9))
    # bottom margin reserves room for the figure caption so the stats box
    # and the caption cannot overlap.
    # PANEL REMOVED — the whole-volume before/after |v| histogram.
    #
    # It pooled every signal voxel inside R₀, which on the Circle of Willis is
    # ~99% parenchymal noise centred near 6 cm/s. The two means differed by
    # 0.01 cm/s (6.56 → 6.54) and the "before" histogram sat entirely behind
    # the "after" one, so the reader saw a single distribution and learned
    # nothing from it.
    #
    # More importantly the null result was guaranteed by construction and so
    # carried no information about correctness: Λ deviations change sign
    # across the FOV, so the correction is close to mean-zero over any large
    # volume whether or not the implementation is right. A panel that looks
    # identical when the code works and when it is broken is not evidence.
    # The one real fact in it is stated in the caption instead.
    #
    # The freed column lets the statistics box span BOTH rows. That box is the
    # densest panel on the sheet and it now carries the direction-error block
    # as well, so it needs the height: sized for a single row its auto-fitted
    # font bottomed out at the 5.5 pt floor.
    gs  = gridspec.GridSpec(2, 4, hspace=0.38, wspace=0.40, bottom=0.11,
                            width_ratios=[1.0, 1.0, 1.0, 1.30])

    # Central axial slice (or thin-slab MIP) at peak systole.
    #
    # The Circle of Willis is a THREE-DIMENSIONAL ring: the ACAs, ICA termini,
    # M1 segments, posterior communicating arteries, basilar tip and PCAs span
    # roughly 10-15 mm in z. At sub-millimetre slice spacing no single axial
    # slice contains the whole structure, which is why a single-slice |v| panel
    # always looks like scattered fragments. Projecting a thin slab
    # (DICOM_FIG05_SLAB slices either side, maximum intensity) assembles the
    # ring in one panel, which is the standard 4D-flow presentation.
    # DICOM_FIG05_SLAB = -1 projects the entire volume, which is what a
    # conventional PC-MRA does and the only way the whole Circle of Willis —
    # ICAs rising from below, vertebrals and basilar from behind — appears in
    # one panel. Any finite slab necessarily cuts part of the tree away.
    if int(DICOM_FIG05_SLAB) < 0:
        _slab = Nz
        _z0, _z1 = 0, Nz
    else:
        _slab = max(0, int(DICOM_FIG05_SLAB))
        _z0 = max(0, iz_show - _slab)
        _z1 = min(Nz, iz_show + _slab + 1)
    # How much of the volume's flow is actually on screen. Computed for
    # BOTH display modes, before the branch, because the figure caption
    # below consumes it unconditionally. Defining it only in the
    # single-slice branch raised
    #   UnboundLocalError: cannot access local variable '_shown_frac'
    # on every MIP-mode dataset — the whole in_vivo_head preset — so
    # fig05 failed there while the phantom and pipe presets worked.
    _shown_frac = None
    try:
        _fv_all = (absd_4d[..., t_sys] > 0) if absd_4d is not None else None
        if _fv_all is not None and _fv_all.any():
            _n_tot = float(_fv_all.sum())
            _shown_frac = (100.0 * float(_fv_all[:, :, _z0:_z1].sum()) / _n_tot
                           if _slab > 0 else
                           100.0 * float(_fv_all[:, :, iz_show].sum()) / _n_tot)
    except Exception:
        _shown_frac = None

    if _slab > 0:
        sl_m = mag_m_4d[:, :, _z0:_z1, t_sys].max(axis=2)
        sl_c = mag_c_4d[:, :, _z0:_z1, t_sys].max(axis=2)
        sl_d = absd_4d[:, :, _z0:_z1, t_sys].mean(axis=2)
        _panel_loc = ("full-volume MIP" if (_z0, _z1) == (0, Nz)
                      else f"slab MIP z={_z0}-{_z1-1}")
        _dloc     = ("full-volume mean" if (_z0, _z1) == (0, Nz)
                     else f"slab mean z={_z0}-{_z1-1}")
    else:
        sl_m = mag_m_4d[:, :, iz_show, t_sys]   # (Nx, Ny)
        sl_c = mag_c_4d[:, :, iz_show, t_sys]
        sl_d = absd_4d[:, :, iz_show, t_sys]
        # HOW MUCH OF THE FLOW IS ON SCREEN.
        #
        # A single axial plane through a phantom whose channels run
        # through-plane shows only cross-sections, so the reader cannot tell
        # from the panels whether the analysis saw the whole flow. It did --
        # every statistic in this figure and in run_config.json is computed
        # over the full volume and all cardiac phases -- but the DISPLAY is
        # one slice of Nz. State the share explicitly rather than leaving the
        # reader to assume the panels are the dataset.
        #
        # A full-volume MIP is not the fix here, which is why the phantom and
        # pipe presets use a single plane: a maximum over Nz projections
        # raises the noise floor by ~sqrt(2*ln(Nz)) sigma, and at Nz = 88 that
        # is ~3.0 sigma. With sigma ~3 cm/s the projected noise reaches ~9
        # cm/s, above the vessel threshold, so the MIP would look like flow
        # everywhere. The in-vivo preset can afford it because the vessel tree
        # is bright and spread across the slab; a phantom with one bright
        # channel and a large static bath cannot.
        # KEEP THIS SHORT. It is embedded in a panel title, and the panels
        # are narrow: the earlier wording ("N% of flow voxels shown —
        # statistics use the full volume") ran wider than the axes and
        # collided with the neighbouring panel's title, so the two overlapped
        # and neither was legible. matplotlib does not clip or wrap titles.
        # The long-form explanation belongs in the figure footnote, which has
        # the full width of the canvas.
        _panel_loc = (f"slice z={iz_show} of {Nz}"
                      + ("" if _shown_frac is None else
                         f", {_shown_frac:.1f}% of flow"))
        _dloc     = f"z={iz_show}"
        if _shown_frac is not None:
            print(f"  [fig05] Display slice holds {_shown_frac:.1f}% of the "
                  f"volume's flow voxels; all statistics are full-volume "
                  f"({Nz} slices x {Nt} phases).")
    # Apply the signal mask to the slice panels: grey out non-signal
    # (background-noise) voxels so the anatomy/flow stands out when a
    # magnitude mask is provided. Without a mask, all voxels are shown.
    if mask is not None:
        _m2d = mask[:, :, _z0:_z1].any(axis=2) if _slab > 0 else mask[:, :, iz_show]
        # Panels 3 and 4 (the signed difference and the |Δv| map) go through
        # _m2d, not through _keep3d, so the coherence gate has to be applied
        # here as well — otherwise the static compartment is removed from the
        # velocity panels but still painted on the correction panels, which is
        # worse than not gating at all: the same object appears present in one
        # half of the figure and absent in the other.
        if flow_gate is not None and flow_gate.shape == mask.shape:
            _fg2 = (flow_gate[:, :, _z0:_z1].any(axis=2) if _slab > 0
                    else flow_gate[:, :, iz_show])
            _m2d = _m2d & _fg2
        sl_m = np.where(_m2d, sl_m, np.nan)
        sl_c = np.where(_m2d, sl_c, np.nan)
        sl_d = np.where(_m2d, sl_d, np.nan)

    # ── Correction panels: flow voxels only ──────────────────────────────
    # The velocity panels keep the whole object, because seeing the phantom
    # (or the head) is what gives the reader spatial context. The CORRECTION
    # panels are cut down to voxels that actually flow: elsewhere |Δv| is the
    # correction applied to rectified noise, which is a real number but not a
    # meaningful one, and there is far more of it than there is flow. Left in,
    # it sets the colour scale and buries the structure.
    _diff2d = None
    if DICOM_FIG05_DIFF_ON_FLOW_ONLY and vessel_thr is not None:
        try:
            _thr_v = (np.asarray(vessel_thr, dtype=np.float32)
                      if np.ndim(vessel_thr) == 3 else float(vessel_thr))
            _ves3 = mag_m_3d > _thr_v
            if mask is not None:
                _ves3 = _ves3 & mask
            if flow_gate is not None and flow_gate.shape == _ves3.shape:
                _ves3 = _ves3 & flow_gate
            _diff2d = (_ves3[:, :, _z0:_z1].any(axis=2) if _slab > 0
                       else _ves3[:, :, iz_show])
            sl_d = np.where(_diff2d, sl_d, np.nan)
            print(f"  [fig05] Correction panels restricted to {int(_ves3.sum()):,} "
                  f"flow voxels "
                  f"(of {int(mask.sum()) if mask is not None else _ves3.size:,} "
                  "signal voxels); static fluid and noise excluded.")
        except Exception as _exc:
            print(f"  [fig05] Flow-only restriction skipped ({_exc}).")
            _diff2d = None

    # Overlay floor. In parenchyma vx, vy and vz are pure noise and
    # |v| = sqrt(vx²+vy²+vz²) rectifies that noise to several cm/s, so a floor
    # set as a fraction of the display max lands INSIDE the noise and paints
    # the whole brain. Use the measured noise floor instead: sigma estimated
    # from the Maxwell median of the masked speeds, cut at N*sigma.
    # A maximum-intensity projection over N slices reports the LARGEST of N
    # noise samples, not a typical one, so the effective noise floor rises with
    # slab thickness. The expected maximum of N draws grows as sigma*sqrt(2 lnN),
    # so the floor is inflated by that amount — otherwise a thick slab paints
    # noise maxima across the whole brain.
    _n_proj = max(1, _z1 - _z0)
    _mip_inflation = float(np.sqrt(2.0 * np.log(_n_proj))) if _n_proj > 1 else 0.0

    def _project(field):
        """Collapse a 3-D field over the display slab the same way as the data."""
        if field is None or np.ndim(field) == 0:
            return field
        return (field[:, :, _z0:_z1].max(axis=2) if _slab > 0
                else field[:, :, iz_show])

    # Compare per voxel FIRST, then project the decision. Projecting the
    # threshold itself with max() forces every voxel in a column to clear the
    # WORST (lowest-SNR) slice in that column, which silently discards genuine
    # vessels sitting in high-SNR slices. The keep-decision is what should be
    # projected, not the bar.
    _keep3d = None
    if vessel_thr is not None and np.ndim(vessel_thr) == 3:
        try:
            _thr3 = np.asarray(vessel_thr, dtype=np.float32)
            _sig3 = (np.asarray(vessel_sigma, dtype=np.float32)
                     if vessel_sigma is not None and np.ndim(vessel_sigma) == 3
                     else _thr3 / max(DICOM_VESSEL_SIGMA_MULT, 1e-9))
            _bar3 = _thr3 + _sig3 * _mip_inflation
            _spd3 = mag_m_4d[:, :, :, t_sys]
            _keep3 = _spd3 > _bar3
            if mask is not None:
                _keep3 = _keep3 & mask
            # Temporal-coherence gate — removes STATIC fluid compartments
            # (e.g. the reference water sphere beside a flow phantom) whose
            # rectified noise otherwise paints the panel with speckle that
            # looks like slow flow. See DICOM_FLOW_COHERENCE_MIN.
            if flow_gate is not None and flow_gate.shape == _keep3.shape:
                _keep3 = _keep3 & flow_gate
            # Drop isolated voxels before projecting (see DICOM_FIG05_MIN_BLOB).
            if DICOM_FIG05_MIN_BLOB and DICOM_FIG05_MIN_BLOB > 1:
                try:
                    from scipy import ndimage as _ndi
                    _n_before_blob = int(_keep3.sum())
                    _lab, _nlab = _ndi.label(_keep3)
                    if _nlab > 0:
                        _sz = _ndi.sum(_keep3, _lab, index=range(1, _nlab + 1))
                        _good = np.zeros(_nlab + 1, bool)
                        _good[1:][_sz >= DICOM_FIG05_MIN_BLOB] = True
                        _keep3 = _good[_lab]
                        print(f"  [fig05] Speckle filter: {_nlab:,} components, "
                              f"kept those >= {DICOM_FIG05_MIN_BLOB} voxels "
                              f"({_n_before_blob:,} -> {int(_keep3.sum()):,} "
                              "voxels shown)")
                except Exception as _exc:
                    print(f"  [fig05] Speckle filter skipped ({_exc}).")
            _keep3d = (_keep3[:, :, _z0:_z1].any(axis=2) if _slab > 0
                       else _keep3[:, :, iz_show])
            _floor_cm = float(np.nanmedian(_bar3[mask]) * 100) if mask is not None \
                else float(np.nanmedian(_bar3) * 100)
            _floor_map_cm = 0.0          # decision already made in 3-D
            _floor_src = (f"{DICOM_VESSEL_SIGMA_MULT:g}·σ_v(r) per-voxel, "
                          "applied per slice before projection "
                          f"(median {_floor_cm:.0f} cm/s)")
        except Exception as _exc:
            print(f"  [fig05] per-slice thresholding failed ({_exc}); "
                  "falling back to projected threshold.")
            _keep3d = None

    if _keep3d is not None:
        pass
    elif vessel_thr is not None:
        _thr_2d = _project(vessel_thr)
        _sig_2d = _project(vessel_sigma) if vessel_sigma is not None else None
        if _sig_2d is None:
            _sig_2d = (np.asarray(_thr_2d)
                       / max(DICOM_VESSEL_SIGMA_MULT, 1e-9))
        # Floor is a MAP when the noise is spatially varying, so scalp and
        # orbit voxels are held to a much higher bar than brain parenchyma.
        _floor_map_cm = (np.asarray(_thr_2d) + np.asarray(_sig_2d)
                         * _mip_inflation) * 100.0
        _floor_cm = float(np.nanmedian(_floor_map_cm))
        _floor_src = (f"{DICOM_VESSEL_SIGMA_MULT:g}·σ_v(r) per-voxel"
                      + (f" + {_mip_inflation:.1f}σ MIP inflation"
                         if _n_proj > 1 else "")
                      + f", median {_floor_cm:.0f} cm/s")
    else:
        # Fall back to estimating sigma from the UNPROJECTED time-averaged
        # speed — never from the MIP, whose distribution is not Maxwell.
        _finite = mag_m_3d[mask] if mask is not None else mag_m_3d.ravel()
        _finite = _finite[np.isfinite(_finite)]
        if _finite.size > 100:
            _sig_cm = float(np.median(_finite) / 1.5382) * 100.0
            _min_cm = (DICOM_VESSEL_MIN_VENC_FRAC * float(venc) * 100.0
                       if venc else 0.0)
            _floor_cm = max(DICOM_VESSEL_SIGMA_MULT * _sig_cm, _min_cm) \
                + _sig_cm * _mip_inflation
            _floor_map_cm = _floor_cm
            _floor_src = f"{DICOM_VESSEL_SIGMA_MULT:g}σ global noise floor (σ={_sig_cm:.1f})"
        else:
            # Sparse-mask fallback. NOTE: `vmax` is not defined until ~20
            # lines below, so referring to it here raised NameError and
            # crashed fig05 whenever the mask held <= 100 finite voxels.
            # Derive the floor from the volume itself instead.
            _finite_all = mag_m_3d[np.isfinite(mag_m_3d)]
            _vol_max_cm = (float(_finite_all.max()) * 100.0
                           if _finite_all.size else 0.0)
            _floor_cm = 0.30 * _vol_max_cm
            _floor_map_cm = _floor_cm
            _floor_src = "30% of volume max (sparse-mask fallback)"
    # Colour range from the SURVIVING voxels only. Scaling to the raw slab
    # maximum lets a single low-SNR noise spike (which can reach sqrt(3)*VENC)
    # set the range and crush the real vessels into the bottom of the colormap.
    _both = np.concatenate([sl_m.ravel(), sl_c.ravel()]) * 100
    if _keep3d is not None:
        _sel = np.concatenate([np.where(_keep3d, sl_m, np.nan).ravel(),
                               np.where(_keep3d, sl_c, np.nan).ravel()]) * 100
        _keep = _sel[np.isfinite(_sel)]
    else:
        _keep = _both[np.isfinite(_both) &
                      (_both > (np.nanmedian(_floor_map_cm)
                                if np.ndim(_floor_map_cm) else _floor_map_cm))]
    if DICOM_FIG05_VMAX_CMS is not None:
        vmax = float(DICOM_FIG05_VMAX_CMS)
        _vmax_src = "manual (DICOM_FIG05_VMAX_CMS)"
    elif _keep.size > 50:
        # 95th, not 99th, percentile. A handful of aliased / dropout voxels
        # (the log reports 97 voxels above VENC on this dataset) sit far above
        # the true flow; the 99th percentile lets them stretch the scale to
        # ~VENC and compress the real 9-14 cm/s flow into the dark end. The
        # 95th percentile tracks the bulk flow and keeps the vessels visible.
        vmax = float(np.percentile(_keep, 95))
        _vmax_src = "auto, 95th percentile of displayed voxels"
    else:
        vmax = float(np.nanpercentile(_both, 95.0))
        _vmax_src = "auto, 95th percentile"
    vmax = max(vmax, 1.0)

    # Warn when the scale is being set by a small fast minority, since the
    # region of interest is then compressed into the bottom of the colormap
    # and appears featureless.
    if DICOM_FIG05_VMAX_CMS is None and _keep.size > 50:
        _med = float(np.median(_keep))
        if _med < 0.20 * vmax:
            print("  [fig05] NOTE: median displayed speed is "
                  f"{_med:.0f} cm/s but the colour scale reaches "
                  f"{vmax:.0f} cm/s.")
            print("           A small fast region is setting the range and "
                  "the rest is compressed into")
            print(f"           the bottom {100*_med/vmax:.0f}% of the colormap. "
                  "Set DICOM_FIG05_VMAX_CMS "
                  f"(try {max(5*_med, 10):.0f}) to see structure.")

    # Display-floor safety cap. The overlay floor is derived from the local
    # noise (sigma-based) or VENC fraction, both of which can exceed the actual
    # flow when VENC is set far above the true velocity. Cap the DISPLAY floor
    # at 60% of the median masked speed so the overlay never hides the bulk of
    # the flow it is meant to show. Display only — never affects statistics.
    try:
        _med_disp = float(np.nanmedian(m_flat[np.isfinite(m_flat)]) * 100.0)
        if _med_disp > 0 and _floor_cm > 0.60 * _med_disp:
            _floor_cm_old = _floor_cm
            _floor_cm = 0.60 * _med_disp
            if np.ndim(_floor_map_cm):
                _floor_map_cm = np.minimum(_floor_map_cm, _floor_cm)
            else:
                _floor_map_cm = min(float(_floor_map_cm), _floor_cm)
            print(f"  [fig05] Display floor capped {_floor_cm_old:.1f} -> "
                  f"{_floor_cm:.1f} cm/s (was above 60% of the median flow; "
                  f"VENC may be set too high for this data).")
    except Exception:
        pass
    print(f"  [fig05] Velocity overlay floor: {_floor_cm:.1f} cm/s "
          f"({_floor_src})")
    print(f"  [fig05] Panels show {_panel_loc.replace(chr(10), ' — ')}; "
          f"colour scale 0-{vmax:.0f} cm/s "
          f"({_vmax_src})")

    # Clinical-style velocity-on-anatomy overlay. When a magnitude volume is
    # available, blood-flow speed (turbo) is overlaid only on the flow voxels
    # (|v| above ~30% of the display max) on top of the grayscale magnitude, so
    # velocity appears on the vessels/anatomy. Otherwise the masked |v| map is
    # shown.
    _anat = None
    if mag_vol is not None and getattr(mag_vol, "ndim", 0) == 3 \
            and 0 <= iz_show < mag_vol.shape[2]:
        _anat = (mag_vol[:, :, _z0:_z1].max(axis=2) if _slab > 0
                 else mag_vol[:, :, iz_show])

    # NOTE: all panels in this figure use origin="upper" (radiological
    # convention, matching a DICOM viewer) so the figure is directly
    # comparable to scanner screenshots. This is a DISPLAY choice only —
    # voxel coordinates, Λ and the correction are unaffected. All panels
    # must use the SAME origin so the velocity overlay stays registered
    # to the magnitude background.
    # ── PC-MRA gate ──────────────────────────────────────────────────────
    # pcmra = normalised magnitude x time-averaged speed. Computed on the FULL
    # time series (not the systolic frame) because averaging collapses the
    # noise tail ~2x while pulsatile flow survives; then projected over the
    # display slab exactly like the velocity panels so the two stay registered.
    _gate2d = None
    if (DICOM_FIG05_PCMRA_GATE and DICOM_FIG05_PCMRA_GATE > 0
            and mag_vol is not None and getattr(mag_vol, "ndim", 0) == 3
            and mag_vol.shape == mag_m_3d.shape):
        try:
            _mn = float(np.nanmax(mag_vol))
            if _mn > 0:
                _pcmra = (mag_vol / _mn) * mag_m_3d        # (Nx,Ny,Nz)
                _pc2d = (_pcmra[:, :, _z0:_z1].max(axis=2) if _slab > 0
                         else _pcmra[:, :, iz_show])
                _ref = float(np.nanpercentile(
                    _pcmra[mask] if mask is not None else _pcmra, 99.9))
                _gthr = DICOM_FIG05_PCMRA_GATE * _ref
                _gate2d = _pc2d > _gthr
                print("  [fig05] PC-MRA gate: keeping "
                      f"{100*_gate2d.mean():.2f}% of the panel "
                      f"(gate = {DICOM_FIG05_PCMRA_GATE:g} x p99.9)")
        except Exception as _exc:
            print(f"  [fig05] PC-MRA gate skipped ({_exc}).")

    # 2-D projection of the coherence gate, for the panels that do not go
    # through _keep3d (the difference map and the |v_cor - v_meas| map).
    _flow2d = None
    if flow_gate is not None and flow_gate.shape == mag_m_3d.shape:
        _flow2d = (flow_gate[:, :, _z0:_z1].any(axis=2) if _slab > 0
                   else flow_gate[:, :, iz_show])
        print(f"  [fig05] Flow-coherence gate: {100*float(flow_gate.mean()):.2f}% "
              "of the volume kept (static compartments removed)")

    def _draw_velocity(ax, sl, title):
        ax.set_facecolor("0.85")
        if _anat is not None:
            ax.imshow(_anat.T, origin="upper", cmap="gray", aspect="auto")
            if _keep3d is not None:
                _keep_v = (~np.isnan(sl)) & _keep3d
            else:
                _keep_v = (~np.isnan(sl)) & (sl * 100 > _floor_map_cm)
            if _gate2d is not None:
                _keep_v = _keep_v & _gate2d
            if _flow2d is not None:
                _keep_v = _keep_v & _flow2d
            vshow = np.where(_keep_v, sl * 100, np.nan)
            im = ax.imshow(vshow.T, origin="upper", cmap="turbo",
                           vmin=0, vmax=vmax, aspect="auto")
        else:
            im = ax.imshow(sl.T * 100, origin="upper", cmap="turbo",
                           vmin=0, vmax=vmax, aspect="auto")
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("x [voxels]", fontsize=8)
        ax.set_ylabel("y [voxels]", fontsize=8)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Panel (b) — Measured
    ax1 = fig.add_subplot(gs[0, 0])
    _draw_velocity(ax1, sl_m,
                   f"|v| measured  [cm/s]\n({_panel_loc}, t={t_sys}, "
                   f"> {_floor_cm:.0f} cm/s)")

    # Panel (c) — Corrected
    ax2 = fig.add_subplot(gs[0, 1])
    _draw_velocity(ax2, sl_c,
                   f"|v| corrected  [cm/s]\n({_panel_loc}, t={t_sys}, "
                   f"> {_floor_cm:.0f} cm/s)")

    # Panel (c2) — Signed difference (corrected − measured), tightly stretched
    # The acquired-data correction is only ~1 cm/s near isocenter, invisible against
    # the 0–120 cm/s scale of panels (b)/(c). Zooming the colour range to a
    # few cm/s makes its SPATIAL STRUCTURE visible: a diverging map where
    # red = the correction sped the flow up, blue = slowed it down. A coherent,
    # smoothly varying pattern (not random speckle) that mirrors the Λ map
    # (the Λ map, saved as fig01) is the acquired-data evidence the correction works.
    ax_d = fig.add_subplot(gs[0, 2])
    sl_diff = (sl_c - sl_m) * 100.0                         # cm/s, signed
    if mask is not None:
        sl_diff = np.where(_m2d, sl_diff, np.nan)
    if _diff2d is not None:
        sl_diff = np.where(_diff2d, sl_diff, np.nan)
    dlim = max(float(np.nanpercentile(np.abs(sl_diff), 95)), 0.5)
    cmap_d = plt.cm.RdBu_r.copy()
    cmap_d.set_bad("0.92")                                  # NaN → light grey
    imd = ax_d.imshow(sl_diff.T, origin="upper", cmap=cmap_d,
                      vmin=-dlim, vmax=dlim, aspect="auto")
    ax_d.set_title("Corrected − Measured |v|  [cm/s]\n"
                   f"(stretched ±{dlim:.1f} cm/s — reveals structure)",
                   fontsize=9)
    ax_d.set_xlabel("x [voxels]", fontsize=8); ax_d.set_ylabel("y [voxels]", fontsize=8)
    plt.colorbar(imd, ax=ax_d, fraction=0.046, pad=0.04)
    if lambda_corr_rel is not None or lambda_corr_r is not None:
        # Primary: relative correction vs Λ map (speed factor removed).
        # Secondary in smaller print: the absolute form, for continuity.
        _ann = []
        if lambda_corr_rel is not None:
            _ann.append(f"corr(|Δv|/|v|, ‖Λ−I‖) = {lambda_corr_rel:+.2f}")
        if lambda_corr_r is not None:
            _ann.append(f"abs (speed-confounded): {lambda_corr_r:+.2f}")
        ax_d.text(0.03, 0.97, "\n".join(_ann),
                  transform=ax_d.transAxes, fontsize=8, va="top", ha="left",
                  color="black",
                  bbox=dict(boxstyle="round", facecolor="white",
                            edgecolor="0.6", alpha=0.85))

    # Panel (d) — Correction magnitude map
    ax3 = fig.add_subplot(gs[1, 0]); ax3.set_facecolor("0.85")
    im3 = ax3.imshow(sl_d.T * 100, origin="upper", cmap="plasma",
                     vmin=0, vmax=max(float(np.nanpercentile(sl_d*100, 99)), 0.1),
                     aspect="auto")
    _flow_note = " — flow voxels only" if _diff2d is not None else ""
    ax3.set_title(f"|v_cor − v_meas|  [cm/s]\n(correction magnitude, {_dloc}"
                  f"{_flow_note})",
                  fontsize=9)
    ax3.set_xlabel("x [voxels]", fontsize=8); ax3.set_ylabel("y [voxels]", fontsize=8)
    plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

    # Panel (e) — Per-slice correction profile
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(per_slice_delta, np.arange(Nz), color="#d62728", lw=1.5)
    ax4.set_xlabel("Mean |Δv|  [cm/s]", fontsize=9)
    ax4.set_ylabel("Slice index (z)", fontsize=9)
    ax4.set_title(f"Correction magnitude\nper slice (time-averaged, {_ps_basis})",
                  fontsize=9)
    ax4.axvline(delta_mean, color="gray", ls="--", lw=1, alpha=0.7,
                label=f"mean {delta_mean:.2f} cm/s")
    ax4.legend(fontsize=8); ax4.grid(True, alpha=0.3)
    ax4.invert_yaxis()   # superior slices at top

    # Panel (g) — Correction magnitude vs distance from isocenter
    # GNL vanishes at isocenter and grows with radius; a rising curve here is
    # the physical signature that the correction is doing gradient-nonlinearity
    # work rather than adding noise.
    ax_r = fig.add_subplot(gs[1, 2])
    if Xd is not None and Yd is not None and Zd is not None:
        r_mm = np.sqrt(Xd**2 + Yd**2 + Zd**2) * 1000.0        # (Nx,Ny,Nz) [mm]
        rr = r_mm[mask]
        dd = absd_3d[mask] * 100.0                            # cm/s
        if rr.size > 20:
            nb = 12
            # Bin over the radius range the data actually covers. Starting the
            # edges at r = 0 wasted ~3 of 12 bins on empty space for an
            # off-isocentre phantom (its nearest voxel is ~26 mm out), which
            # threw away a quarter of the radial resolution.
            edges   = np.linspace(float(rr.min()), float(rr.max()), nb + 1)
            edges[-1] += 1e-6                     # keep the outermost voxel
            centres = 0.5 * (edges[:-1] + edges[1:])
            mean_d  = np.full(nb, np.nan)
            p90_d   = np.full(nb, np.nan)
            # A bin of 5 voxels gives a mean with no meaning and a visible
            # spurious dip at the outer edge; require a usable sample.
            _min_bin = 25
            for i in range(nb):
                sel = (rr >= edges[i]) & (rr < edges[i + 1])
                if sel.sum() >= _min_bin:
                    mean_d[i] = dd[sel].mean()
                    p90_d[i]  = np.percentile(dd[sel], 90)
            ax_r.plot(centres, mean_d, "-o", color="#7b3294", lw=1.6, ms=4,
                      label="mean |Δv|")
            ax_r.plot(centres, p90_d, "--", color="#c2a5cf", lw=1.2,
                      label="90th percentile")
            ax_r.set_xlabel("Distance from isocenter  r [mm]", fontsize=9)
            ax_r.set_ylabel("Correction |Δv|  [cm/s]", fontsize=9)
            # State what is shown, not what the reader should conclude. GNL
            # vanishing at r = 0 is a property of the model, not something
            # this panel demonstrates — an off-isocentre phantom never
            # samples small r, so the curve cannot be seen to reach zero.
            ax_r.set_title(
                "Correction |Δv| vs distance from isocentre\n"
                f"(sampled range r = {rr.min():.0f}–{rr.max():.0f} mm; "
                "Λ → I at r → 0 by construction, not sampled here)",
                fontsize=8)
            ax_r.legend(fontsize=7); ax_r.grid(True, alpha=0.3)
        else:
            ax_r.axis("off")
            ax_r.text(0.5, 0.5, "insufficient voxels\nfor radial profile",
                      ha="center", va="center", transform=ax_r.transAxes,
                      fontsize=9, color="grey")
    else:
        ax_r.axis("off")
        ax_r.text(0.5, 0.5, "radial profile unavailable\n(coordinates not provided)",
                  ha="center", va="center", transform=ax_r.transAxes,
                  fontsize=9, color="grey")

    # Panel (f) — Statistics text box
    ax5 = fig.add_subplot(gs[0:2, 3])
    ax5.axis("off")
    # ── DIRECTION-ERROR block ────────────────────────────────────────────
    # The task description names errors in "both the magnitude AND direction
    # of encoded velocities". |Δv| covers magnitude; nothing in this figure
    # reported direction, even though direction_error_stats() had already
    # written the numbers into vessel_stats. They are shown here.
    #
    # theta is the angle between v_measured and v_corrected — the rotation the
    # correction APPLIES, not an error against a ground truth (there is none
    # for acquired data). Its diagnostic value is the RADIAL TREND: velocity
    # noise rotates the vector randomly and averages out, whereas the gradient
    # field rotates it systematically and more strongly further from
    # isocentre. A positive corr(theta, r) is therefore the model signature,
    # and is reported alongside the percentiles rather than the mean alone.
    _dir_lines = []
    if vessel_stats and vessel_stats.get("dir_err_median_deg") is not None:
        _dm  = vessel_stats.get("dir_err_median_deg")
        _dp95 = vessel_stats.get("dir_err_p95_deg")
        _dp99 = vessel_stats.get("dir_err_p99_deg")
        _dmax = vessel_stats.get("dir_err_max_deg")
        _dn   = vessel_stats.get("dir_err_n_samples")
        _dcr  = vessel_stats.get("corr_direrr_vs_radius")
        _dfl  = vessel_stats.get("dir_err_speed_floor_cm_s")
        _dir_lines = [
            "── ENCODING-DIRECTION ERROR ──",
            (f"  angle(v_meas, v_corr), |v| > {_dfl:.0f} cm/s"
             if _dfl is not None else "  angle(v_meas, v_corr)"),
            f"  n = {_dn:,} analysis voxels × phases" if _dn else "",
            f"  median {_dm:.3f}°   p95 {_dp95:.3f}°"
            if _dp95 is not None else f"  median {_dm:.3f}°",
        ]
        if _dp99 is not None and _dmax is not None:
            _dir_lines.append(f"  p99    {_dp99:.3f}°   max {_dmax:.3f}°")
        if _dcr is not None:
            _dir_lines.append(f"  corr(θ, radius) = {_dcr:+.3f}")
        # Radial shells: the trend matters more than any single percentile,
        # so the shell medians are listed when they exist.
        _dprof = vessel_stats.get("dir_err_radial_profile") or []
        _dprof = [q for q in _dprof if q.get("n", 0) >= 50]
        if len(_dprof) >= 2:
            _dir_lines.append("  r [mm]      θ med    n")
            for _q in _dprof:
                # direction_error_stats stores r_hi_mm = None for the final,
                # OPEN-ENDED shell (">120 mm"). Its own console printer handles
                # that; this one formatted it with :3.0f and raised
                #   TypeError: unsupported format string passed to
                #   NoneType.__format__
                # which killed fig05 outright. The bin only appears when the
                # data actually reaches past the last fixed edge AND the shell
                # holds >= 50 samples, so the Circle-of-Willis and cardiac
                # phantom runs never hit it — the pipe rig, sitting 91 mm off
                # isocentre, did.
                _rng = (f"{_q['r_lo_mm']:3.0f}-{_q['r_hi_mm']:3.0f}"
                        if _q.get("r_hi_mm") is not None
                        else f"   >{_q['r_lo_mm']:3.0f}")
                _dir_lines.append(
                    f"  {_rng}   "
                    f"{_q['theta_median_deg']:6.3f}°  {_q['n']:7,}")
        _dir_lines = [ln for ln in _dir_lines if ln != ""] + [""]

    # ── Analysis-set block ───────────────────────────────────────────────
    # The header must describe the set that was actually measured. With the
    # velocity gate disabled the set is the magnitude/brain mask, NOT vessels,
    # and calling it "VESSEL VOXELS ONLY" would be self-contradictory next to
    # a printed threshold of 0 cm/s. The noise-consistency fraction is shown
    # in the same block so the relative-correction figure can be read
    # correctly without going back to run_config.json:
    #   per-voxel |Δv|/|v| is SCALE-INVARIANT, so it measures the local
    #   deviation of Λ from identity and stays valid over noise voxels;
    #   the ABSOLUTE |Δv| in cm/s is diluted by them and is not a
    #   blood-flow number when the gate is off.
    def _vessel_block(vs):
        _gate_off = bool(vs.get("velocity_gate_disabled"))
        _nc = vs.get("noise_consistency") or {}
        _frac = _nc.get("frac_at_or_below_noise")
        if _gate_off:
            head = ["── ANALYSIS SET: magnitude/brain mask ──",
                    f"  no velocity gate, n={vs['n_vessel_voxels']:,}"]
        else:
            head = ["── VESSEL VOXELS ONLY (headline) ──",
                    f"  threshold {vs['vessel_threshold_cm_s']:.0f} cm/s, "
                    f"n={vs['n_vessel_voxels']:,}"]
        body = [
            f"  mean |v| = {vs['vmeas_mean_cm_s']:.1f} cm/s",
            f"  corr. mean {vs['delta_mean_cm_s']:.3f}  "
            f"med {vs['delta_median_cm_s']:.3f}",
            f"  corr. p95  {vs['delta_p95_cm_s']:.3f} cm/s",
            f"  mean rel. correction = {vs['pct_signal_corrected']:.1f}%",
        ]
        tail = []
        if _frac is not None:
            tail = [f"  {100*_frac:.0f}% of set at/below noise "
                    f"({_nc.get('noise_mean_speed_cm_s', 0):.1f} cm/s)"]
            if _gate_off and _frac > 0.5:
                tail.append("  -> rel.% valid (scale-free);")
                tail.append("     abs cm/s NOT a blood number")
        return head + body + tail + [""]

    lines = [
        "DICOM Correction Results",
        "─" * 32,
        f"Dataset: {coil}",
        f"Shape: {Nx}×{Ny}×{Nz} vox × {Nt} phases",
        "",
        # Three decimals, and the difference stated explicitly.
        #
        # At two decimals these two lines read 4.89 and 4.89, which looks like
        # the correction did nothing. It is not a bug and the fix is not to
        # manufacture a difference: the near-equality is the expected result.
        # Lambda deviations change SIGN across the FOV (see the butterfly
        # pattern in the Lambda figure), so summed over a large volume the
        # correction cancels: it REDISTRIBUTES velocity rather than rescaling
        # it. The size of the change is |Δv| below — a per-voxel magnitude
        # that does not cancel — not the difference of these two means.
        f"Mean |v|  before: {vmeas_mean:7.3f} cm/s",
        f"          after : {vcorr_mean:7.3f} cm/s",
        f"          diff  : {vcorr_mean - vmeas_mean:+7.3f} cm/s "
        f"({100.0*(vcorr_mean-vmeas_mean)/max(vmeas_mean,1e-9):+.2f}%)",
        "  near-zero by design (Λ changes sign)",
        "",
        "Correction |Δv| (all signal):",
        f"  mean {delta_mean:.3f}  median {delta_med:.3f}",
        f"  p95  {delta_p95:.3f} cm/s",
        # mean|Δv| / mean|v| — a MEAN RELATIVE CORRECTION, not a voxel
        # fraction; the old "% of signal modified" label invited exactly
        # that misreading.
        f"  mean rel. correction = {pct_change:.1f}%",
        "",
    ] + (_vessel_block(vessel_stats) if vessel_stats else []) + _dir_lines + [
        f"Validation vs Λ map ({_fig_label(2).strip('[]')}):",
        (f"  corr(|Δv|/|v|, ‖Λ−I‖) = {lambda_corr_rel:+.3f}"
         if lambda_corr_rel is not None else
         "  corr(|Δv|/|v|, ‖Λ−I‖) = n/a"),
        (f"  abs (speed-confounded): {lambda_corr_r:+.3f}"
         if lambda_corr_r is not None else
         "  abs (speed-confounded): n/a"),
    ]
    # ── Size the font to the panel, in BOTH directions ───────────────────
    #
    # The first version used fixed line-count thresholds and let a 34-line box
    # overflow into the caption. The second sized on panel HEIGHT alone, which
    # fixed the overflow but left the box unreadable: height is the binding
    # constraint here (the panel holds ~40 short lines), so the font was
    # pinned near the floor while roughly half the panel WIDTH stayed empty.
    #
    # Both limits are now computed and the smaller wins, so the text grows
    # until it runs out of room in one direction or the other. For a monospace
    # face the advance width is ~0.60 em, so a line of L characters needs
    # L * 0.60 * fontsize points. The ceiling is 11 pt: past that the box
    # starts to compete with the panel titles for attention.
    fig.canvas.draw()
    _bb    = ax5.get_window_extent(fig.canvas.get_renderer())
    _h_pt  = _bb.height * 72.0 / fig.dpi
    _w_pt  = _bb.width  * 72.0 / fig.dpi
    _n_ln  = max(len(lines), 1)
    _max_c = max((len(ln) for ln in lines), default=1)
    _fs_h  = _h_pt / (_n_ln * 1.32)                 # line height incl. spacing
    _fs_w  = (_w_pt * 0.94) / (_max_c * 0.60)       # 6% margin for the padding
    _fs    = float(max(5.5, min(11.0, min(_fs_h, _fs_w))))
    print(f"  [DICOM] stats box: {_n_ln} lines x {_max_c} chars → "
          f"{_fs:.1f} pt (height allows {_fs_h:.1f}, width allows {_fs_w:.1f})")
    ax5.text(0.02, 0.99, "\n".join(lines),
             transform=ax5.transAxes, fontsize=_fs, linespacing=1.15,
             va="top", ha="left", family="monospace",
             bbox=dict(boxstyle="round", pad=0.35, facecolor="lightyellow",
                       edgecolor="goldenrod", alpha=0.85))

    # Dataset label for the title: use the explicit global override if set,
    # otherwise derive it from the output directory name so the figure title
    # always matches the dataset actually being corrected.
    _dcm_num_pre = FIG_RENUMBER.get(19, 19)
    if DICOM_DATASET_LABEL and DICOM_DATASET_LABEL.strip():
        _label = DICOM_DATASET_LABEL.strip()
    else:
        _stem = Path(output_dir).name
        for _pfx in ("Output_", "output_", "OUTPUT_", "Output ", "output "):
            if _stem.startswith(_pfx):
                _stem = _stem[len(_pfx):]
        _label = _stem.replace("_", " ").strip() or "DICOM dataset"

    fig.suptitle(
        f"DICOM Correction Summary — {_label}  |  {coil}\n"
        "Comparing v_measured vs v_corrected  "
        "(no independent ground truth — the Λ-map correlation is an "
        "implementation consistency check, not a validation)",
        fontsize=10, fontweight="bold")

    # ONE LINE. A thesis figure carries a short note; the discussion belongs
    # in the caption underneath it in the document.
    #
    # It must also describe the display mode that was ACTUALLY used. The
    # previous wording said "ONE slice of {Nz}" unconditionally, so a
    # full-volume MIP — the in_vivo_head default — was described as "ONE slice
    # of 52 holding 100.0% of the volume's flow voxels", which contradicts
    # itself: a projection through every slice is not one slice, and the
    # 100% is the giveaway.
    if _slab > 0 and (_z0, _z1) == (0, Nz):
        _what = "a full-volume maximum-intensity projection"
    elif _slab > 0:
        _what = (f"a MIP over slices {_z0}–{_z1 - 1} of {Nz}"
                 + ("" if _shown_frac is None else
                    f" ({_shown_frac:.0f}% of flow voxels)"))
    else:
        _what = (f"slice {iz_show} of {Nz}"
                 + ("" if _shown_frac is None else
                    f" ({_shown_frac:.1f}% of flow voxels)"))
    fig.text(0.5, 0.015,
             f"|v| = √(vx²+vy²+vz²) from the phase data; magnitude masks "
             f"background only. Panels show {_what}; all statistics use the "
             "full volume and every phase. Radiological convention (origin "
             "upper) — display only.",
             ha="center", va="bottom", fontsize=9, color="0.25", wrap=True)


    out_p = Path(output_dir) / f"fig{_dcm_num_pre:02d}_dicom_correction.png"
    if ENABLED_FIGS is None or 19 in ENABLED_FIGS:
        fig.savefig(out_p, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  [DICOM] Diagnostic figure saved → {out_p.name}")
    else:
        plt.close(fig)



# ─────────────────────────────────────────────────────────────────────────────
#  Encoding-direction error metric
# ─────────────────────────────────────────────────────────────────────────────
def write_isocentre_marker(path: str, arm_mm: float = 80.0) -> str:
    """
    Three orthogonal LPS axis arms crossing at (0, 0, 0), as an ASCII legacy
    VTK POLYDATA file, for loading in ParaView alongside velocity_field.vtk.

    The magnet isocentre is where Lambda = I by construction, so the whole
    claim of the method is that the correction vanishes there and grows away
    from it. A reader can only judge that if the point is visible in the
    figure. The marker also makes the slab offset explicit: for a head
    acquisition the cross typically sits outside the slab, which IS the
    geometry, not an error.

    Coordinates match velocity_field.vtk: DICOM patient LPS in millimetres,
    origin at isocentre. 'axis' scalar: 0 = x (+ = Left), 1 = y
    (+ = Posterior), 2 = z (+ = Superior).
    """
    a = float(arm_mm)
    pts = [(-a, 0, 0), (a, 0, 0), (0, -a, 0), (0, a, 0), (0, 0, -a), (0, 0, a)]
    with open(path, "w") as f:
        f.write("# vtk DataFile Version 3.0\n"
                "isocentre marker, DICOM patient LPS mm\nASCII\n"
                "DATASET POLYDATA\n")
        f.write(f"POINTS {len(pts)} float\n")
        for p in pts:
            f.write(f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f}\n")
        f.write("LINES 3 9\n2 0 1\n2 2 3\n2 4 5\n")
        f.write(f"POINT_DATA {len(pts)}\nSCALARS axis float 1\n"
                "LOOKUP_TABLE default\n0\n0\n1\n1\n2\n2\n")
    return path


def direction_error_stats(v_meas, v_corr, vessel, Xd, Yd, Zd,
                          speed_min_m_s=0.0, stats=None, verbose=True):
    """
    Angle between the measured and the corrected velocity vector, per vessel
    voxel and per cardiac phase:

        theta = arccos( (v_meas . v_corr) / (|v_meas| |v_corr|) )

    (theta_max in the coil deviation summary is a different quantity: a
    property of the gradient system over the full +/-220 mm FOV, not of the
    acquired data.)

    WHAT THIS IS AND IS NOT. theta is the rotation the correction APPLIES,
    not an error measured against a ground truth — acquired data has none.
    Since v_corr = Lambda^-T v_meas exactly, theta is a DETERMINISTIC function
    of Lambda and the direction of v_meas, so its rise with radius is
    guaranteed by the construction of Lambda whatever the data says. It is
    therefore an IMPLEMENTATION CONSISTENCY CHECK -- it confirms Lambda was
    evaluated at the right positions and applied with the right convention --
    and NOT independent validation of the field model. fig05's caption states
    it that way; keep this docstring, the stats box and the caption in
    agreement.

    It remains the useful direction-side statistic because the thesis topic
    names errors in "both the magnitude and direction of encoded velocities":
    |Delta v| and the radial |Delta v|/|v| table cover magnitude, theta covers
    direction, measured on the acquired slab rather than at coil level.

    Parameters
    ----------
    v_meas, v_corr : (3, Nx, Ny, Nz) or (3, Nx, Ny, Nz, Nt), m/s
    vessel         : (Nx, Ny, Nz) bool, the vessel mask already in use
    Xd, Yd, Zd     : (Nx, Ny, Nz) voxel coordinates in METRES, patient LPS
                     with the origin at isocentre (as used for Lambda)
    speed_min_m_s  : ignore samples below this |v_meas| — direction is
                     meaningless where the vector is pure noise
    stats          : optional dict updated in place (vessel_stats), so the
                     values reach run_config.json

    Returns
    -------
    dict of summary values (also merged into `stats` when given).
    """
    vessel = np.asarray(vessel, bool)
    n_ves = int(vessel.sum())
    if n_ves < 20:
        if verbose:
            print("  [DICOM] Direction-error metric skipped "
                  f"(only {n_ves} vessel voxels).")
        return {}

    # index the mask first so nothing large is copied
    vm = np.asarray(v_meas)[:, vessel]
    vc = np.asarray(v_corr)[:, vessel]
    if vm.ndim == 2:
        vm = vm[..., None]
    if vc.ndim == 2:
        vc = vc[..., None]
    vm = vm.astype(np.float64)
    vc = vc.astype(np.float64)

    nm = np.linalg.norm(vm, axis=0)
    nc = np.linalg.norm(vc, axis=0)
    ok = (nm > max(speed_min_m_s, 1e-9)) & (nc > 1e-9)
    if int(ok.sum()) < 20:
        if verbose:
            print("  [DICOM] Direction-error metric skipped "
                  "(too few samples above the speed floor).")
        return {}

    cos = np.einsum('ijk,ijk->jk', vm, vc) / np.maximum(nm * nc, 1e-30)
    theta = np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))
    th = theta[ok]

    out = {
        "dir_err_mean_deg":   float(th.mean()),
        "dir_err_median_deg": float(np.median(th)),
        "dir_err_p95_deg":    float(np.percentile(th, 95)),
        "dir_err_p99_deg":    float(np.percentile(th, 99)),
        "dir_err_max_deg":    float(th.max()),
        "dir_err_n_samples":  int(ok.sum()),
        "dir_err_speed_floor_cm_s": float(speed_min_m_s * 100),
    }

    r_vox = np.sqrt(np.asarray(Xd) ** 2 + np.asarray(Yd) ** 2 +
                    np.asarray(Zd) ** 2)[vessel] * 1000.0
    r_all = np.broadcast_to(r_vox[:, None], theta.shape)[ok]

    prof = []
    # The lowest edge is the smallest radius the data ACTUALLY samples, not 0.
    # A slab that never comes closer than 32 mm to isocentre was previously
    # reported in a bin labelled "0-40 mm", which reads as though isocentre
    # itself had been measured.
    _r_lo = float(np.nanmin(np.sqrt(np.asarray(Xd) ** 2 + np.asarray(Yd) ** 2 +
                                    np.asarray(Zd) ** 2)[np.asarray(vessel, bool)])
                  * 1000.0)
    _edges = [np.floor(_r_lo), 40, 60, 80, 100, 120, 1e9]
    _edges = [e for i, e in enumerate(_edges)
              if i == 0 or e > _edges[0]]
    for _lo, _hi in zip(_edges[:-1], _edges[1:]):
        _sel = (r_all >= _lo) & (r_all < _hi)
        if int(_sel.sum()) < 50:
            continue
        prof.append({
            "r_lo_mm": float(_lo),
            "r_hi_mm": None if _hi > 1e8 else float(_hi),
            "theta_median_deg": float(np.median(th[_sel])),
            "theta_p95_deg": float(np.percentile(th[_sel], 95)),
            "n": int(_sel.sum()),
        })
    out["dir_err_radial_profile"] = prof

    _fin = np.isfinite(r_all) & np.isfinite(th)
    if int(_fin.sum()) > 100 and float(np.std(r_all[_fin])) > 0:
        out["corr_direrr_vs_radius"] = float(
            np.corrcoef(r_all[_fin], th[_fin])[0, 1])

    if verbose:
        print("\n  [DICOM] ── Encoding-direction error "
              "angle(v_meas, v_corr) ──")
        print(f"           samples: {out['dir_err_n_samples']:,} "
              "(vessel voxels × cardiac phases, "
              f"|v| > {out['dir_err_speed_floor_cm_s']:.0f} cm/s)")
        print(f"           mean   = {out['dir_err_mean_deg']:.3f}°")
        print(f"           median = {out['dir_err_median_deg']:.3f}°")
        print(f"           p95    = {out['dir_err_p95_deg']:.3f}°")
        print(f"           p99    = {out['dir_err_p99_deg']:.3f}°")
        print(f"           max    = {out['dir_err_max_deg']:.3f}°")
        if "corr_direrr_vs_radius" in out:
            print("           corr(θ, radius) = "
                  f"{out['corr_direrr_vs_radius']:+.3f}")
        if prof:
            print("             r [mm]      θ med °     θ p95 °        n")
            for _p in prof:
                _lbl = (f"{_p['r_lo_mm']:.0f}-{_p['r_hi_mm']:.0f}"
                        if _p['r_hi_mm'] else f">{_p['r_lo_mm']:.0f}")
                print(f"             {_lbl:>7}    "
                      f"{_p['theta_median_deg']:8.3f}    "
                      f"{_p['theta_p95_deg']:8.3f}   {_p['n']:>8,}")
        print("           theta is the rotation the correction APPLIES, a "
              "deterministic function of\n           Lambda -- so its rise "
              "with radius is built in. This is an implementation\n"
              "           consistency check (Lambda evaluated at the right "
              "positions, applied with\n           the right convention), "
              "NOT independent validation of the field model.")

    if stats is not None:
        stats.update(out)
    return out


class _CliArgs:
    """Stand-in for PathSelectorGUI when the paths come from the command line.

    Exposes exactly the attributes main() reads from the GUI object, so the
    headless and interactive paths share one code path below. tkinter is not
    imported or touched in this mode, which is what lets the toolbox run on a
    Linux server or in CI where no display exists.
    """

    def __init__(self, a):
        self.grad_file      = a.grad
        self.dicom_dir      = a.dicom
        self.mag_dir        = a.magnitude
        self.output_dir     = a.out
        self.dataset_preset = a.preset
        self.sim_mode       = a.sim_mode
        self.flow_pattern   = a.flow_pattern
        self.run_dicom      = bool(a.dicom)
        self.venc_override  = a.venc
        self.direction_order = a.direction_order
        self.no_axis_flags   = bool(a.no_axis_flags)
        self.direction_permute = a.direction_permute
        self.direction_signs   = a.direction_signs
        self.mask_frac       = a.mask_frac
        self.fig05_slice     = a.fig05_slice
        self.mode1_pulsatile = bool(a.mode1_pulsatile)

    def launch(self):
        return True


def _parse_permute_arg(text: str) -> str:
    """argparse type: a permutation of 'xyz' (case-insensitive)."""
    t = str(text).strip().lower()
    if sorted(t) != ["x", "y", "z"]:
        import argparse as _ap
        raise _ap.ArgumentTypeError(
            f"must be a permutation of 'xyz' (e.g. 'yxz'), got {text!r}")
    return t


def _parse_signs_arg(text: str) -> Tuple[int, int, int]:
    """argparse type: three comma-separated values of +1 / -1."""
    import argparse as _ap
    parts = [p for p in re.split(r"[,\s]+", str(text).strip()) if p]
    try:
        vals = tuple(int(p) for p in parts)
    except ValueError:
        vals = ()
    if len(vals) != 3 or any(v not in (1, -1) for v in vals):
        raise _ap.ArgumentTypeError(
            f"must be three values of +1/-1 such as '1,1,-1', got {text!r}")
    return vals  # type: ignore[return-value]


def _parse_cli(argv=None):
    """Command-line interface. Returns None when the GUI should be used."""
    import argparse
    ap = argparse.ArgumentParser(
        prog="4D_Flow_GNL_Correction",
        description=(
            "Gradient nonlinearity correction for phase-contrast / 4D flow "
            "MRI (generalised velocity reconstruction, Markl et al. 2003). "
            "Run with no arguments to use the graphical file picker."),
        epilog=(
            "example:\n"
            "  python 4D_Flow_GNL_Correction_coh5.py \\\n"
            "      --grad coeff_SC72.grad.txt \\\n"
            "      --dicom /data/phantom/velocity --magnitude /data/phantom/magnitude \\\n"
            "      --out results/phantom --preset ideal_phantom \\\n"
            "      --venc 1.50"),
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grad", metavar="FILE",
                    help="gradient coefficient file (.grad / .txt)")
    ap.add_argument("--dicom", metavar="DIR",
                    help="DICOM directory holding the velocity series "
                         "(omit to run the simulation only)")
    ap.add_argument("--magnitude", metavar="DIR", default=None,
                    help="DICOM directory holding the magnitude series, used "
                         "for the signal mask")
    ap.add_argument("--out", metavar="DIR", help="output directory")
    ap.add_argument("--preset", default=DATASET_PRESET,
                    choices=["manual", "in_vivo_head", "ideal_phantom",
                             "pipe_flow"],
                    help=f"masking preset (default: {DATASET_PRESET})")
    ap.add_argument("--snr-db", dest="snr_db", type=float, default=None,
                    metavar="DB",
                    help="simulation SNR in dB; default matches the noise "
                         "floor measured in the data (the run reports it)")
    ap.add_argument("--sim-venc", dest="sim_venc", type=float, default=None,
                    metavar="M_S",
                    help="simulation VENC in m/s; default follows --venc")
    ap.add_argument("--sim-nt", dest="sim_nt", type=int, default=None,
                    metavar="N",
                    help="simulation cardiac phases; default follows the "
                         "acquisition")
    ap.add_argument("--venc", type=float, default=None, metavar="M_S",
                    help="VENC of the acquired data in m/s; overrides "
                         "auto-detection")
    ap.add_argument("--mode1-pulsatile", dest="mode1_pulsatile",
                    action="store_true",
                    help="legacy Mode 1: scale the Poiseuille profile by "
                         "0.85 + 0.15*waveform (85-100%% of v_max) instead of "
                         "steady flow. Only for reproducing synthetic results "
                         "made before v1.0")
    ap.add_argument("--sim-mode", dest="sim_mode", type=int, default=2,
                    choices=[1, 2], help="1 = Poiseuille, 2 = pulsatile tube")
    ap.add_argument("--flow-pattern", dest="flow_pattern", default="helical",
                    choices=["straight", "helical"])
    ap.add_argument("--grid-n", dest="grid_n", type=int, default=None,
                    help="simulation voxels per axis; keep ODD so isocentre "
                         "is sampled")
    ap.add_argument("--direction-order", dest="direction_order", default=None,
                    metavar="XYZ",
                    help="which patient axis each of the three stored velocity "
                         "frames encodes, e.g. 'yzx' = frame0->y, frame1->z, "
                         "frame2->x. Applied AFTER the dataset preset, which "
                         "otherwise resets DICOM_DIRECTION_ORDER to None, so "
                         "this is the only reliable way to set it per run")
    ap.add_argument("--fig05-slice", dest="fig05_slice", type=int, default=None,
                    metavar="N",
                    help="show this slice index in fig05 instead of the automatic "
                         "flow-content pick (0-based, in the DICOM's own slice order). "
                         "Use when the automatic choice lands on a corrupted or "
                         "artefact-contaminated slice.")
    ap.add_argument("--mask-frac", dest="mask_frac", type=float, default=None,
                    metavar="F",
                    help="signal-mask threshold as a fraction (0-1) of the magnitude "
                         "maximum, applied AFTER the dataset preset. The right value "
                         "depends on the acquisition: it must sit above the background "
                         "and BELOW the dimmest part of the object, or the mask clips "
                         "the lumen where the signal is weak")
    ap.add_argument("--direction-permute", dest="direction_permute",
                    type=_parse_permute_arg, default=None, metavar="XYZ",
                    help="permute the LOADED velocity components for this run, e.g. "
                         "'yxz' swaps vx and vy. Applied after --direction-order and "
                         "before --direction-signs. Default: none. Derive it for each "
                         "acquisition; never reuse one from another series")
    ap.add_argument("--direction-signs", dest="direction_signs",
                    type=_parse_signs_arg, default=None, metavar="SX,SY,SZ",
                    help="per-component sign flips for this run, three values of +1/-1, "
                         "e.g. --direction-signs=1,1,-1 (use '=' when the first value is "
                         "negative). Default: none")
    ap.add_argument("--no-axis-flags", dest="no_axis_flags", action="store_true",
                    help="do NOT apply any axis permutation / sign flips, including "
                         "values edited into DICOM_DIRECTION_PERMUTE / _SIGNS in the "
                         "file. Overrides --direction-permute / --direction-signs")
    ap.add_argument("--no-npz", action="store_true",
                    help="skip the (large) dicom_corrected.npz export")
    ap.add_argument("--gui", action="store_true",
                    help="force the graphical picker even with arguments")
    a = ap.parse_args(argv)

    if a.gui or not any((a.grad, a.dicom, a.out)):
        return None                      # interactive mode
    missing = [n for n, v in (("--grad", a.grad), ("--out", a.out)) if not v]
    if missing:
        ap.error("headless mode needs " + " and ".join(missing))
    return a


def main(argv=None):
    np.random.seed(42)

    print("=" * 65)
    print("  4D Flow MRI — Gradient Nonlinearity Correction Simulation")
    print("  Master Thesis — Otto von Guericke University Magdeburg")
    print("=" * 65)

    # ── Path selection: command line, else GUI ────────────────────────────
    _cli = _parse_cli(argv)
    if _cli is not None:
        global DICOM_SAVE_NPZ, _CLI_GRID_N
        global SIM_SNR_DB_OVERRIDE, SIM_VENC_OVERRIDE, SIM_NT_OVERRIDE
        SIM_SNR_DB_OVERRIDE = _cli.snr_db
        SIM_VENC_OVERRIDE   = _cli.sim_venc
        SIM_NT_OVERRIDE     = _cli.sim_nt
        if _cli.no_npz:
            DICOM_SAVE_NPZ = False
        _CLI_GRID_N = _cli.grid_n
        print("  Headless mode (no display required)")
        gui = _CliArgs(_cli)
        ok  = True
    else:
        if not TK_AVAILABLE:
            raise SystemExit(
                "No arguments given and tkinter is unavailable, so the file "
                "picker cannot open.\n"
                "Run headless instead, e.g.\n"
                "  python 4D_Flow_GNL_Correction_coh5.py --grad COEFF.grad "
                "--dicom DIR --out DIR\n"
                "or install tkinter (Debian/Ubuntu: sudo apt install "
                "python3-tk).\n"
                "Use --help for the full option list.")
        gui = PathSelectorGUI()
        ok  = gui.launch()
    if not ok:
        print("[EXIT] User cancelled.")
        sys.exit(0)

    grad_file    = gui.grad_file
    output_dir   = gui.output_dir
    dicom_dir    = gui.dicom_dir

    # ── Output-collision guard ───────────────────────────────────────────
    # Every run writes the SAME filenames: fig01..fig05, run_config.json,
    # dicom_corrected.npz, velocity_field.vtk. Pointing two datasets at one
    # folder therefore destroys the first set of results with no warning —
    # the one way separate runs really can affect each other, since Python
    # globals reset with every process and the presets are symmetric.
    # If this folder already holds results from a DIFFERENT dicom_dir, write
    # into a named subfolder instead of overwriting. Re-running the SAME
    # dataset still overwrites in place, which is what you want while
    # iterating.
    try:
        _prev = Path(output_dir) / "run_config.json"
        if _prev.is_file():
            with open(_prev, "r", encoding="utf-8") as _f:
                _old = json.load(_f)
            _old_dcm = str(_old.get("runtime", {}).get("dicom_dir") or "")
            _new_dcm = str(dicom_dir or "")
            if _old_dcm and _new_dcm and \
                    Path(_old_dcm).resolve() != Path(_new_dcm).resolve():
                _tag = (Path(_new_dcm).name or "dataset")
                _tag = re.sub(r"[^A-Za-z0-9._-]+", "_", _tag).strip("_")
                _sub = Path(output_dir) / f"{_tag}_{gui.dataset_preset}"
                _sub.mkdir(parents=True, exist_ok=True)
                print("\n  !! This output folder already holds results from a "
                      "different dataset:")
                print(f"       existing : {_old_dcm}")
                print(f"       this run : {_new_dcm}")
                print(f"     Writing into a subfolder instead so the earlier "
                      f"results survive:\n       {_sub}\n")
                output_dir = str(_sub)
    except Exception as _oe:
        print(f"  [output] Collision check skipped "
              f"({type(_oe).__name__}: {_oe}) — files may be overwritten.")

    sim_mode     = gui.sim_mode
    flow_pattern = gui.flow_pattern
    run_dicom    = gui.run_dicom
    # GUI-selected magnitude / reference folders override the module defaults.
    global DICOM_MAGNITUDE_DIR, DICOM_VENC_OVERRIDE_MS
    global DICOM_DIRECTION_ORDER, DICOM_DIRECTION_PERMUTE, DICOM_DIRECTION_SIGNS
    global DICOM_MAGNITUDE_MASK_FRAC, DICOM_MASK_FRAC_EXPLICIT
    if gui.mag_dir:
        DICOM_MAGNITUDE_DIR = gui.mag_dir
    # Apply the dataset preset chosen in the dialog. This re-sets the whole
    # group of coupled options every time, so a phantom run cannot leave the
    # brain mask off for a subsequent Circle-of-Willis run, or vice versa.
    _apply_dataset_preset(gui.dataset_preset, announce=True)

    # The GUI is the authority when it was shown: a number means that VENC,
    # a BLANK field means auto-detect. Without the else-branch a value left in
    # the settings above would survive a blank field and be applied to a
    # different dataset — e.g. the steady-flow phantom's VENC silently reused
    # on the cardiac-cycle phantom, which scales every velocity wrongly and is
    # not detectable from the output.
    DICOM_VENC_OVERRIDE_MS = gui.venc_override      # None => auto-detect
    # Per-run axis settings from the command line. They are applied AFTER the
    # dataset preset on purpose: the preset resets DICOM_DIRECTION_ORDER, so a
    # value edited at the top of the file never reaches a run that uses one.
    if getattr(gui, "direction_order", None):
        DICOM_DIRECTION_ORDER = str(gui.direction_order).strip().lower()
        print(f"  Direction order: '{DICOM_DIRECTION_ORDER}' — set on the command "
              "line (frame0, frame1, frame2 -> these patient axes)")
    if getattr(gui, "mask_frac", None) is not None:
        _mf = float(gui.mask_frac)
        if not (0.0 < _mf < 1.0):
            raise ValueError(f"--mask-frac must be between 0 and 1, got {_mf}")
        DICOM_MAGNITUDE_MASK_FRAC = _mf
        DICOM_MASK_FRAC_EXPLICIT = True
        print(f"  Mask fraction  : {_mf:.0%} of the magnitude maximum — set on the "
              "command line (overrides the preset)")
    # Axis permutation / signs: explicit per-run values from the command line.
    # They are applied AFTER the dataset preset (which may reset the flags) and
    # bypass the scope guard, because the user named them for THIS dataset.
    global DICOM_DIRECTION_SCOPE, AXIS_FLAGS_SOURCE
    _cli_perm  = getattr(gui, "direction_permute", None)
    _cli_signs = getattr(gui, "direction_signs", None)
    if _cli_perm or _cli_signs:
        DICOM_DIRECTION_PERMUTE = _cli_perm
        DICOM_DIRECTION_SIGNS   = tuple(_cli_signs) if _cli_signs else None
        DICOM_DIRECTION_SCOPE   = None
        AXIS_FLAGS_SOURCE = "cli" if isinstance(gui, _CliArgs) else "dialog"
        print(f"  Axis flags     : permute={DICOM_DIRECTION_PERMUTE or 'none'}  "
              f"signs={DICOM_DIRECTION_SIGNS or 'none'} — set "
              + ("on the command line" if AXIS_FLAGS_SOURCE == "cli"
                 else "in the dialog") + " for this run")
    elif DICOM_DIRECTION_PERMUTE or DICOM_DIRECTION_SIGNS:
        AXIS_FLAGS_SOURCE = "file"
        print(f"  Axis flags     : permute={DICOM_DIRECTION_PERMUTE or 'none'}  "
              f"signs={DICOM_DIRECTION_SIGNS or 'none'} — EDITED INTO THE FILE "
              f"(scope guard: {DICOM_DIRECTION_SCOPE!r}). Prefer "
              "--direction-permute / --direction-signs")
    else:
        AXIS_FLAGS_SOURCE = "none"
        print("  Axis flags     : none — velocity components used as loaded")
    if getattr(gui, "no_axis_flags", False):
        DICOM_DIRECTION_PERMUTE = None
        DICOM_DIRECTION_SIGNS = None
        AXIS_FLAGS_SOURCE = "none (--no-axis-flags)"
        print("  Axis flags     : DICOM_DIRECTION_PERMUTE / _SIGNS DISABLED for this run "
              "(--no-axis-flags)")
    if DICOM_VENC_OVERRIDE_MS is None:
        print("  VENC          : auto-detect from filename/header "
              "(blank in the dialog)")
    else:
        print(f"  VENC          : {DICOM_VENC_OVERRIDE_MS:.2f} m/s "
              f"({DICOM_VENC_OVERRIDE_MS*100:.0f} cm/s) — entered in the dialog")
    # The GUI is the authority when it was shown: a number means that slice,
    # a BLANK field means automatic. Without the else-branch a value baked into
    # the settings above would survive a blank field and be applied to a
    print(f"\n  Gradient file : {grad_file}")
    print(f"  Output dir    : {output_dir}")
    if dicom_dir:
        print(f"  DICOM dir     : {dicom_dir}")
    print(f"  Simulation    : Mode {sim_mode}")
    print(f"  Flow pattern  : {flow_pattern}\n")

    # ── Parse gradient coefficient file ──────────────────────────────────
    print("[Stage 1] Parsing gradient coefficient file …")
    grad_coeffs = GradientFieldParser.parse(grad_file)

    # ── Auto-detect main field strength B0 from coil identifier ──────────
    # IMPORTANT: at the BMMR laboratory there are TWO distinct 7 T systems
    # (corrected per supervisor feedback):
    # · SC72CD → 7 T (the 7 T Plus system, denoted "7T+")
    # · IMPULSE/Athena → 7 T (the 7 T Terra.X Impulse Edition, "7Tx")
    # Only the Skyra is a 3 T clinical scanner. The two 7 T systems have
    # different gradient-coil geometries and must not be conflated. B0 is
    # recorded per coil so the coil-vs-header consistency check can flag a
    # .grad file that does not belong to the scanner that produced the data.
    _coil_lc  = (grad_coeffs.coil_name + " " +
                 Path(grad_coeffs.filepath).name).lower()
    if any(k in _coil_lc for k in ("impulse", "athena", "terra")):
        detected_B0 = 7.0
        scanner_tag = "7T Terra.X Impulse Edition (7Tx)"
    elif any(k in _coil_lc for k in ("sc72",)):
        detected_B0 = 7.0
        scanner_tag = "7T Plus (7T+, SC72CD coil)"
    elif any(k in _coil_lc for k in ("skyra", "davinci", "3t")):
        detected_B0 = 3.0
        scanner_tag = "3T (Skyra clinical)"
    elif "7t" in _coil_lc:
        detected_B0 = 7.0
        scanner_tag = "7T (generic — coil identifier ambiguous)"
    else:
        detected_B0 = 3.0
        scanner_tag = "3T (default — coil not recognised)"
    print(f"  [B0-detect] Coil → {scanner_tag}   |   B0 = {detected_B0:.1f} T")

    # ── Simulation config ─────────────────────────────────────────────────
    cfg = SimulationConfig(flow_pattern=flow_pattern, B0=detected_B0)
    if getattr(gui, "mode1_pulsatile", False):
        cfg.mode1_pulsatile = True
        print("  Mode 1        : LEGACY mildly pulsatile profile "
              "(--mode1-pulsatile)")
    if _CLI_GRID_N:
        cfg.grid_N = int(_CLI_GRID_N)
    # VENC is known now (dialog or --venc); Nt only after the DICOM loads, so
    # it is matched again there. Overrides win over both.
    _apply_sim_overrides(cfg, acq_venc_ms=DICOM_VENC_OVERRIDE_MS)


    # ── Self-tests ────────────────────────────────────────────────────────
    print("\n[Stage 1b] Running pipeline self-tests …")
    all_ok = run_self_tests(cfg, grad_coeffs=grad_coeffs)
    if not all_ok:
        print("[WARN] One or more self-tests failed — "
              "results may be unreliable. Review output above.")

    # ── Build spatial grid ────────────────────────────────────────────────
    print(f"\n[Stage 2] Building {cfg.grid_N}³ spatial grid "
          f"(±{cfg.r_max_m*1000:.0f} mm) …")
    N       = cfg.grid_N
    r_max   = cfg.r_max_m
    ax_lin  = np.linspace(-r_max, r_max, N)
    xv_mm   = ax_lin * 1000.0
    X, Y, Z = np.meshgrid(ax_lin, ax_lin, ax_lin, indexing="ij")

    # ── Compute Λ matrix ──────────────────────────────────────────────────
    print(f"\n[Stage 3] Computing Λ(r) distortion matrix on {N}³ grid …")
    t0          = time.perf_counter()
    evaluator   = GradientFieldEvaluator(grad_coeffs)
    # Forward CS phase convention setting from cfg
    evaluator.set_cs_phase_omit(cfg.cs_phase_omit)
    evaluator.set_sh_normalization(
        getattr(cfg, "sh_normalization", "schmidt"))
    Lambda      = evaluator.compute_lambda(X, Y, Z, eps=cfg.eps_m)
    d_i         = evaluator.strength_deviation(Lambda)
    theta       = evaluator.angular_deviation_deg(Lambda)
    kappa       = evaluator.condition_number_field(Lambda)   # numerical stability
    # Mask of voxels inside the calibration sphere r ≤ R0;
    # results outside this mask are extrapolated and unreliable.
    # margin = eps excludes the shell where the central difference
    # straddles the r = R0 clamp (see inside_r0_mask).
    inside_r0   = evaluator.inside_r0_mask(X, Y, Z, margin=cfg.eps_m)
    n_inside    = int(inside_r0.sum())
    n_total     = inside_r0.size
    pct_inside  = 100.0 * n_inside / n_total
    print(f"  Λ computation: {time.perf_counter()-t0:.1f} s")
    print(f"  κ(Λ) range:    [{kappa.min():.2f}, {np.percentile(kappa, 99):.1f}]"
          f"  median={np.median(kappa):.2f}")
    print(f"  [r≤R0 mask]    {n_inside}/{n_total} voxels inside calibration "
          f"sphere ({pct_inside:.1f}% — {100-pct_inside:.1f}% extrapolated)")

    # ── Init engines ──────────────────────────────────────────────────────
    plotter     = PlottingEngine(output_dir, coil_name=grad_coeffs.coil_name)
    val_engine  = ValidationEngine()
    corrector   = VelocityCorrector(cfg)
    # Nearest sampled plane to the requested display offset (see
    # FIG_CORONAL_SLICE_MM). Falls back to the centre plane if the offset
    # lies outside the grid.
    iy          = int(np.argmin(np.abs(ax_lin - FIG_CORONAL_SLICE_MM / 1000.0)))
    # B0 taken from the coil auto-detection

    # ── Configuration logging (reproducibility — ) ─────────
    print("\n[Stage 3b] Writing run configuration to JSON …")
    config_log = {
        "timestamp_utc":   datetime.now(timezone.utc).isoformat(),
        "run_info": {
            "title":      ("Open-Source Preprocessing for Gradient "
                           "Nonlinearity Correction in 4D Flow MRI"),
            "institution": ("Otto von Guericke University Magdeburg, "
                            "Faculty of Electrical Engineering and IT"),
            "script":      Path(__file__).name,
        },
        "gradient_coil": {
            "filepath":    grad_coeffs.filepath,
            "coil_name":   grad_coeffs.coil_name,
            "R0_m":        grad_coeffs.R0,
            "max_degree":  grad_coeffs.max_degree,
            "lnorm":       grad_coeffs.lnorm,
            "n_coefficients": {ax: len(v)
                               for ax, v in grad_coeffs.coeffs.items()},
            # Scanner detection
            "B0_T":           cfg.B0,
            "scanner_tag":    scanner_tag,
        },
        "simulation_config": asdict(cfg),
        "runtime": {
            "sim_mode":    sim_mode,
            "flow_pattern": flow_pattern,
            "dicom_dir":   dicom_dir,
            "output_dir":  output_dir,
            "dataset_preset":        gui.dataset_preset,
            "dicom_direction_signs": (list(DICOM_DIRECTION_SIGNS)
                                      if DICOM_DIRECTION_SIGNS else None),
            "dicom_direction_permute": DICOM_DIRECTION_PERMUTE,
            "dicom_direction_scope":   DICOM_DIRECTION_SCOPE,
            "axis_flags_source":       AXIS_FLAGS_SOURCE,
            "dicom_geometric_slice_spacing": bool(DICOM_GEOMETRIC_SLICE_SPACING),
            "dicom_direction_order":   DICOM_DIRECTION_ORDER,
            "random_seed": 42,
            "numpy_version":      np.__version__,
            "python_executable":  sys.executable,
        },
        # ── Λ summary, restricted to the calibration sphere ──────────────
        # These numbers characterise the COIL and are quoted in the thesis, so
        # they must obey the same rule as every other metric in this file:
        # r <= R0 only. The spherical-harmonic expansion is undefined outside
        # it and _bz_field clamps the radial factor there, so the values in
        # that shell are an artefact of the clamp, not field data. On the
        # default 97^3 grid over +/-220 mm that shell is 29.8% of the voxels
        # and it visibly moves the statistics: on an earlier 48^3 run,
        # unmasked this block reported
        # kappa_median 1.539 / d_x_min 0.756 / theta_z_max 30.96 deg, against
        # the correct 1.410 / 0.787 / 29.89 deg. fig03 was already masked, so
        # the JSON and the figure disagreed.
        # The unmasked values are kept under *_whole_grid as a diagnostic only.
        "lambda_summary": {
            "mask":          "r <= R0 - eps (calibration sphere)",
            "n_voxels":      int(inside_r0.sum()),
            "n_voxels_total": int(inside_r0.size),
            "kappa_median":  float(np.median(kappa[inside_r0])),
            "kappa_p95":     float(np.percentile(kappa[inside_r0], 95)),
            "kappa_p99":     float(np.percentile(kappa[inside_r0], 99)),
            "kappa_max":     float(np.min([kappa[inside_r0].max(), 1e12])),
            "d_i_min":       {ax: float(d_i[i][inside_r0].min())
                              for i, ax in enumerate(["x", "y", "z"])},
            "d_i_max":       {ax: float(d_i[i][inside_r0].max())
                              for i, ax in enumerate(["x", "y", "z"])},
            "theta_max_deg": {ax: float(theta[i][inside_r0].max())
                              for i, ax in enumerate(["x", "y", "z"])},
            "whole_grid_diagnostic_do_not_quote": {
                "why": ("includes r > R0 where the radial factor is clamped; "
                        "Lambda there is an extrapolation artefact, NOT the "
                        "identity — measured median |Lambda - I| = 0.29 "
                        "outside R0 on the default grid, because the clamp "
                        "removes only the RADIAL derivative and the angular "
                        "dependence survives."),
                "kappa_median":  float(np.median(kappa)),
                "kappa_p95":     float(np.percentile(kappa, 95)),
                "kappa_p99":     float(np.percentile(kappa, 99)),
                "d_i_min":       {ax: float(d_i[i].min())
                                  for i, ax in enumerate(["x", "y", "z"])},
                "theta_max_deg": {ax: float(theta[i].max())
                                  for i, ax in enumerate(["x", "y", "z"])},
            },
        },
        # M7 — self-test outcomes belong in the reproducibility record, not
        # only in the console scrollback. T5 is the non-circular test and is
        # the one the thesis quotes.
        "self_tests": SELF_TEST_RESULTS,
    }
    cfg_path = Path(output_dir) / "run_config.json"
    # Make the partial record reachable from the top-level handler. Without
    # this, an exception anywhere between here and the final write kills the
    # process and run_config.json is left frozen at this early state, with no
    # indication that the run did not finish or why. That happened on the
    # Series 11 phantom run: the file ended after self_tests, and because the
    # DICOM stage has its own except clause the failure could not even be
    # localised from the file. The record now says so itself.
    _RUN_STATE["path"] = cfg_path
    _RUN_STATE["log"] = config_log
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(config_log, f, indent=2, default=str)
    print(f"  → Configuration saved to: {cfg_path.name}")

    # ── Shared figures ────────────────────────────────────────────────────
    print("\n[Stage 4] Generating shared diagnostic figures …")
    # inside_r0 is passed so both figures grey out the clamped extrapolation
    # beyond the calibration sphere instead of displaying it as if it were
    # field data (it is ~31% of the default grid and otherwise sets the
    # colour scale). _y_slice_mm reports where the displayed plane actually
    # sits: an even grid_N has no sample at y = 0.
    _y_slice_mm = float(ax_lin[iy] * 1000.0)
    if abs(FIG_CORONAL_SLICE_MM) > 1e-6:
        print(f"  [fig] Displayed coronal plane: y = {_y_slice_mm:+.2f} mm "
              f"(requested {FIG_CORONAL_SLICE_MM:+.1f} mm). y = 0 is a "
              "symmetry plane where Λ_xy, Λ_yx, Λ_yz and Λ_zy vanish "
              "identically, so an off-centre plane is shown; metrics are "
              "unaffected.")
    elif abs(_y_slice_mm) > 1e-6:
        print(f"  [fig] Displayed coronal plane: y = {_y_slice_mm:+.2f} mm "
              f"(grid_N = {N} is even, so no sample lies at y = 0; "
              "use an odd grid_N to sample isocentre exactly)")
    plotter.plot_lambda_matrix(Lambda, xv_mm, iy,
                               inside_r0=inside_r0, y_mm=_y_slice_mm)
    # Spatially varying VENC(r) —
    plotter.plot_spatially_varying_venc(Lambda, cfg.VENC, xv_mm, iy,
                                        inside_r0=inside_r0,
                                        y_mm=_y_slice_mm)

    # Print coil deviation summary
    # Masked to r ≤ R0, matching lambda_summary in run_config.json and the
    # fig03 percentiles. Unmasked, these lines mixed in the clamped
    # extrapolation shell (~31% of the default grid) and disagreed with both.
    print(f"\n  ── Coil Deviation Summary ({grad_coeffs.coil_name}) ─────────────")
    print(f"  (restricted to the calibration sphere r ≤ R₀, "
          f"{int(inside_r0.sum()):,}/{inside_r0.size:,} voxels)")
    for ci, ax_nm in enumerate(["x", "y", "z"]):
        _d = d_i[ci][inside_r0]
        print(f"  Coil-{ax_nm}: d ∈ [{_d.min():.4f}, {_d.max():.4f}]"
              f"  |  θ_max = {theta[ci][inside_r0].max():.2f}°")

    # ── Simulation runs ───────────────────────────────────────────────────
    modes = ([1, 2] if sim_mode == 3
             else [sim_mode])

    all_results = {}
    for mode in modes:
        res = run_simulation_mode(
            mode, X, Y, Z, Lambda, d_i, theta, xv_mm,
            cfg, grad_coeffs, plotter, val_engine, corrector,
            inside_r0=inside_r0,         # restrict metrics to r ≤ R0
        )
        all_results[mode] = res

    # ── SNR sensitivity sweep ─────────────────────────────────────────────
    # Run once, on the last (most representative) mode. See run_snr_sweep for
    # why a single percentage improvement is not quotable on its own.
    _snr_rows = None
    if RUN_SNR_SWEEP and modes:
        try:
            _snr_rows = run_snr_sweep(
                X, Y, Z, Lambda, cfg, corrector, val_engine,
                inside_r0=inside_r0, snr_list=SNR_SWEEP_DB,
                mode=modes[-1])
        except Exception as _e_sw:
            print(f"  [SNR sweep] skipped: {_e_sw}")

    # ── Optional DICOM pipeline ─────────────
    # When a DICOM directory is provided, the GNL correction (this toolbox's
    # only correction stage) is applied to the loaded real-data volume. Λ is re-evaluated on the DICOM grid because the simulation
    # grid (97³ ±220 mm by default) does not in general coincide with the scanner's
    # acquisition grid. No ground truth is available, so we report
    # "correction magnitude" statistics (|v_cor − v_meas|) rather than
    # RMSE-vs-truth.
    dicom_summary = None
    _layout_frac_over_venc = None
    _mask_recovery = None
    _axis_autoassign = None
    _mask_record = None
    _pipe_extra = None
    # M6 — default so the DICOM summary always carries an axis-flag record,
    # even on code paths that never reach the scope guard.
    _axis_applied = {"permute_applied": False, "signs_applied": False,
                     "scope": DICOM_DIRECTION_SCOPE, "in_scope": None,
                     "reason": "axis-flag block not reached"}
    if run_dicom and PYDICOM_AVAILABLE and dicom_dir:
        print(f"\n[Stage 5] DICOM pipeline:  {dicom_dir}")
        try:
            loader    = DicomLoader()
            dcm_files = loader.find_dicoms(dicom_dir)
            if not dcm_files:
                print("  [DICOM] No .dcm/.IMA files found in directory.")
            else:
                vol, hdr = loader.load_series(dcm_files)
                # vol shape: (3, Nx_d, Ny_d, Nz_d, Nt_d) ← always 5-D now
                v_dcm   = np.asarray(vol, dtype=np.float32)
                if v_dcm.ndim == 4:
                    # Old singleframe loader returned (Nx,Ny,Nz,Nt) with
                    # no component axis. Promote to (1, Nx, Ny, Nz, Nt)
                    # so the rest of the pipeline is uniform.
                    v_dcm = v_dcm[np.newaxis]

                # ── Scope guard for the axis flags ───────────────────────
                # These describe one specific acquisition. Refuse to apply
                # them to any other series (e.g. the in-vivo Circle of
                # Willis), where an unverified swap would be invisible in
                # |v| yet corrupt the Λ⁻ᵀ cross-term mixing.
                _axis_ok = True
                # M6 — the JSON must record what was DONE, not what was
                # CONFIGURED. run_config.json previously listed
                # dicom_direction_permute = "yxz" and signs = (1,1,-1) for a
                # Circle-of-Willis run where the scope guard had (correctly)
                # skipped both, so anyone reproducing from that file would
                # apply an axis swap and a sign flip to head data — silently,
                # because neither changes |v|.
                _axis_applied = {"permute_applied": False,
                                 "signs_applied":   False,
                                 "source":          AXIS_FLAGS_SOURCE,
                                 "scope":           DICOM_DIRECTION_SCOPE,
                                 "in_scope":        None,
                                 "reason":          "no axis flags configured"}
                if DICOM_DIRECTION_PERMUTE or DICOM_DIRECTION_SIGNS:
                    _scope = (str(DICOM_DIRECTION_SCOPE).strip()
                              if DICOM_DIRECTION_SCOPE is not None else None)
                    if _scope == "":
                        _axis_ok = False
                        _axis_applied["reason"] = (
                            "flags disabled: DICOM_DIRECTION_SCOPE = ''")
                        print("  [DICOM] Axis permutation/sign flags disabled "
                              "(DICOM_DIRECTION_SCOPE = '').")
                    elif _scope:
                        _hay = " ".join(str(s) for s in
                                        (dicom_dir or "",
                                         DICOM_DATASET_LABEL or "",
                                         output_dir or "")).lower()
                        _axis_ok = _scope.lower() in _hay
                        _axis_applied["in_scope"] = bool(_axis_ok)
                        # A match that comes only from the output folder name
                        # is not a property of the data: rename the folder and
                        # the component mapping changes. Say so.
                        _in_dicom = _scope.lower() in str(dicom_dir or "").lower()
                        _axis_applied["matched_dicom_path"] = bool(_in_dicom)
                        if _axis_ok and not _in_dicom:
                            print("  [DICOM] WARNING: DICOM_DIRECTION_SCOPE "
                                  f"'{_scope}' was NOT found in the DICOM "
                                  "path — it matched")
                            print("           the output folder or dataset "
                                  "label instead. Renaming that folder would "
                                  "silently")
                            print("           change the velocity component "
                                  "mapping. Point the scope at the DICOM "
                                  "directory.")
                        _axis_applied["reason"] = (
                            f"scope '{_scope}' "
                            + ("matched the dataset path/label"
                               if _axis_ok else
                               "NOT found in the dataset path/label — flags "
                               "skipped, axes used exactly as loaded"))
                        if _axis_ok:
                            print(f"  [DICOM] Axis flags in scope "
                                  f"('{_scope}' matched) — applying the "
                                  "verified component mapping.")
                        else:
                            print()
                            print("  " + "=" * 68)
                            print("  [DICOM] Axis permutation/sign flags NOT "
                                  "applied to this dataset.")
                            print(f"          DICOM_DIRECTION_SCOPE = "
                                  f"'{_scope}' does not match this series, so")
                            print("          the component mapping verified "
                                  "on another acquisition is")
                            print("          deliberately skipped. This "
                                  "dataset is processed with the axes")
                            print("          exactly as the loader assigned "
                                  "them.")
                            print("  " + "=" * 68)
                            print()
                    elif AXIS_FLAGS_SOURCE in ("cli", "dialog"):
                        _axis_applied["reason"] = (
                            f"set for this run ({AXIS_FLAGS_SOURCE}: "
                            "--direction-permute / --direction-signs or the "
                            "dialog fields)")
                        print(f"  [DICOM] Axis flags set for this run "
                              f"({AXIS_FLAGS_SOURCE}) — applying them to this "
                              "series.")
                    else:
                        _axis_applied["reason"] = (
                            "DICOM_DIRECTION_SCOPE = None — flags edited into "
                            "the file applied unconditionally")
                        print("  [DICOM] Axis flags edited into the file applied "
                              "UNCONDITIONALLY (DICOM_DIRECTION_SCOPE = None) — "
                              "verify they belong to this series.")

                # ── Post-load component permutation ──────────────────────
                # Repairs the axis assignment that came out of the loader,
                # whatever route produced it. Runs BEFORE the sign flips so
                # DICOM_DIRECTION_SIGNS is expressed in final x/y/z terms.
                if (_axis_ok and DICOM_DIRECTION_PERMUTE is not None
                        and v_dcm.shape[0] == 3):
                    _ps = str(DICOM_DIRECTION_PERMUTE).strip().lower()
                    if sorted(_ps) != ["x", "y", "z"]:
                        raise ValueError(
                            "DICOM_DIRECTION_PERMUTE must be a permutation of "
                            f"'xyz' (e.g. 'yxz'), got "
                            f"{DICOM_DIRECTION_PERMUTE!r}")
                    _po = ["xyz".index(c) for c in _ps]
                    if _po != [0, 1, 2]:
                        v_dcm = np.ascontiguousarray(v_dcm[_po])
                        _axis_applied["permute_applied"] = True
                        _axis_applied["permute"] = _ps
                        print("  [DICOM] Component axes PERMUTED post-load "
                              f"(DICOM_DIRECTION_PERMUTE = '{_ps}'): "
                              + ", ".join(
                                  f"v{'xyz'[i]}←loaded_{_ps[i]}"
                                  for i in range(3)))
                        print("           This overrides the loaded "
                              "frame→axis assignment. If you have also fixed "
                              "the order upstream via")
                        print("           --direction-order, drop "
                              "--direction-permute — otherwise the swap is "
                              "applied twice.")

                # ── Optional per-component sign flips ────────────────────
                # Applied AFTER any axis permutation done by the loader.
                # See the DICOM_DIRECTION_SIGNS docstring for the ParaView
                # pump-direction verification that justifies setting this.
                if (_axis_ok and DICOM_DIRECTION_SIGNS is not None
                        and v_dcm.shape[0] == 3):
                    _sg = tuple(int(s) for s in DICOM_DIRECTION_SIGNS)
                    if len(_sg) != 3 or any(s not in (-1, 1) for s in _sg):
                        raise ValueError(
                            "DICOM_DIRECTION_SIGNS must be a 3-tuple of ±1, "
                            f"got {DICOM_DIRECTION_SIGNS!r}")
                    for _a in range(3):
                        if _sg[_a] == -1:
                            v_dcm[_a] *= -1.0
                    _axis_applied["signs_applied"] = any(s == -1 for s in _sg)
                    _axis_applied["signs"] = list(_sg)
                    print("  [DICOM] Velocity component signs flipped per "
                          f"DICOM_DIRECTION_SIGNS = {_sg} "
                          f"(negated: {', '.join('v'+ 'xyz'[a] for a in range(3) if _sg[a] == -1) or 'none'}).")

                Nt_d    = v_dcm.shape[-1]
                shape_d = v_dcm.shape[1:-1]          # (Nx, Ny, Nz)
                print(f"  Volume shape: {v_dcm.shape}  |  "
                      f"VENC = {hdr.get('VENC', cfg.VENC)*100:.0f} cm/s  |  "
                      f"Nt = {Nt_d}  |  format = {hdr.get('format','?')}")

                # Does the chosen .grad file belong to this scanner? Checked
                # here, before Lambda is computed from it, so a mismatch is
                # visible before the 5-minute recomputation rather than after.
                DicomLoader.check_coil_matches_scanner(cfg.B0, scanner_tag)

                # ── Is this actually phase data? ──────────────────────────
                # Pointing the tool at a MAGNITUDE series instead of a phase
                # series is an easy mistake — the folders sit side by side and
                # differ only by a number. Nothing downstream would notice:
                # magnitude pixel values run 0..4095, the phase→velocity
                # formula maps those onto 0..VENC, and the result looks like a
                # uniformly fast object. Every statistic would be meaningless
                # while appearing well-formed, so this is checked explicitly.
                _problems = []
                _geom_state = None   # ND / DIS2D / DIS3D — set below if known

                # (a) Header: a series marked magnitude and never phase.
                try:
                    _types = set()
                    _tbl_rel = None
                    for _f in dcm_files[:24]:
                        if _f == "<manual-multidir>":
                            continue
                        _h = pydicom.dcmread(_f, stop_before_pixels=True,
                                             force=True)
                        _types |= {str(x).upper()
                                   for x in getattr(_h, "ImageType", [])}
                        # Siemens private ImageTypeText (0051,1016), e.g.
                        # "p2 P/ND", "p2 M/DIS2D". A research export (MeVisLab,
                        # DirectDicomImport, ...) rewrites the standard
                        # ImageType to DERIVED/SECONDARY/OTHER/<tool> and the
                        # ND / DIS2D / DIS3D token is LOST from it — but the
                        # private tag usually survives. Reading it recovers
                        # the distortion state on exactly the datasets where
                        # the standard tag went "UNKNOWN".
                        try:
                            _itt = _h.get((0x0051, 0x1016), None)
                            if _itt is not None and _itt.value:
                                _types |= {t.strip().upper()
                                           for t in re.split(
                                               r"[\s/]+", str(_itt.value))
                                           if t.strip()}
                        except Exception:
                            pass
                        # Siemens private ImaRelTablePosition (0019,1014).
                        # The gradient-coil frame has its origin at the magnet
                        # ISOCENTRE. The DICOM patient frame coincides with it
                        # only when the table sits at the position the patient
                        # frame was registered against; a non-zero RELATIVE
                        # table position means the volume is offset from
                        # isocentre by that vector and every Lambda would be
                        # evaluated at the wrong radius. This turns the
                        # isocentre assumption into a CHECKED precondition.
                        if _tbl_rel is None:
                            try:
                                _trp = _h.get((0x0019, 0x1014), None)
                                if _trp is not None and _trp.value is not None:
                                    _tbl_rel = [float(x) for x in _trp.value]
                            except Exception:
                                pass

                    # ── Geometric distortion-correction state ────────────
                    # compute_lambda() ASSUMES voxel positions are the TRUE,
                    # vendor-unwarped positions (its docstring cites Markl
                    # 2003 / Lorenz 2014 on this). Siemens writes the state
                    # into ImageType: DIS3D (full 3D unwarp), DIS2D
                    # (in-plane only) or ND (no distortion correction). This
                    # verifies the assumption instead of leaving it implicit.
                    if _types:
                        if "DIS3D" in _types:
                            _geom_state = "DIS3D"
                            print("  [DICOM] ImageType: DIS3D — positions "
                                  "are vendor-unwarped in 3D; matches the "
                                  "Λ position assumption.")
                        elif "DIS2D" in _types:
                            _geom_state = "DIS2D"
                            print("  [DICOM] ImageType: DIS2D — in-plane "
                                  "unwarp only. The SLICE (z) positions may "
                                  "still be distorted; Markl et al. 2003 "
                                  "corrected the slice position separately "
                                  "for exactly this case. Λ is evaluated at "
                                  "slightly wrong z near the FOV edge.")
                        elif "ND" in _types:
                            _geom_state = "ND"
                            print("\n  " + "!" * 68)
                            print("  [DICOM] ImageType: ND — NO vendor "
                                  "distortion correction was applied.")
                            print("          Λ(r) is therefore evaluated at "
                                  "geometrically DISTORTED voxel")
                            print("          positions (compute_lambda "
                                  "assumes unwarped positions). Near")
                            print("          the FOV edge that misplaces the "
                                  "correction by several mm.")
                            print("          Re-export with 3D distortion "
                                  "correction enabled, or accept")
                            print("          this as a stated approximation.")
                            print("  " + "!" * 68 + "\n")
                        else:
                            # QUIET BY DESIGN. This used to be a 12-line
                            # banner demanding the ORIGINAL series be checked
                            # at the console. That demand turned out to be
                            # unnecessary: evaluating Lambda at the displaced
                            # coordinates bounds the induced error at ~2% of
                            # the correction magnitude (9.5% worst case) even
                            # under the worst assumption that NO geometric
                            # correction was applied — on a head-sized slab
                            # that is ~0.03 cm/s against a per-voxel noise
                            # floor of ~4 cm/s, i.e. unmeasurable.
                            #
                            # The limitation is real and stays recorded in
                            # run_config.json under geometry_correction_note,
                            # where it belongs. It does not warrant stopping
                            # the reader on every run.
                            _geom_state = None
                            print("  [DICOM] Neither ImageType nor the Siemens "
                                  "private ImageTypeText (0051,1016) carries "
                                  "an")
                            print("          ND / DIS2D / DIS3D token — "
                                  "geometric correction state UNKNOWN.")
                            print("          Λ is evaluated at the stored "
                                  "coordinates; if they are not vendor-"
                                  "unwarped the induced error is a few % of "
                                  "the correction.")
                            print("          Recorded in run_config.json "
                                  "(geometry_correction_note); state it as a "
                                  "bounded limitation.")

                    if _types and "P" not in _types and "PHASE" not in _types \
                            and ("M" in _types or "MAGNITUDE" in _types):
                        _problems.append(
                            "the DICOM ImageType says MAGNITUDE, not PHASE")
                except Exception:
                    pass

                # (b) Two of the three components identically zero.
                if v_dcm.shape[0] >= 2:
                    _empty = [a for a in range(v_dcm.shape[0])
                              if not np.any(v_dcm[a])]
                    if len(_empty) >= 2:
                        _problems.append(
                            f"{len(_empty)} of {v_dcm.shape[0]} velocity "
                            "components are identically zero")
                elif v_dcm.shape[0] == 1:
                    _problems.append(
                        "only ONE velocity component was found — a 4D flow "
                        "series needs three")

                # NOTE: a third check — mean |v| close to VENC — cannot be
                # applied here. In phase-contrast data the AIR background has
                # uniformly random phase, so its |v| averages ~96% of VENC by
                # construction. Over a whole volume that is mostly background
                # the mean speed is therefore always high, for correct data as
                # much as for wrong data. That test only discriminates once the
                # magnitude mask exists, so it runs later (search for
                # "magnitude-data check").

                if _problems:
                    print()
                    print("  " + "!" * 68)
                    print("  [DICOM] THIS DOES NOT LOOK LIKE PHASE DATA:")
                    for _p in _problems:
                        print(f"            - {_p}")
                    print()
                    print("  A 4D flow phase series contains THREE velocity-")
                    print("  encoding directions. Check the folder: the phase")
                    print("  series usually has 3x as many files as the")
                    print("  magnitude series acquired alongside it.")
                    print()
                    print("  Results computed from magnitude data are")
                    print("  meaningless even though they look well-formed.")
                    print("  " + "!" * 68)
                    if DICOM_STRICT_INPUT_CHECK:
                        raise ValueError(
                            "Input does not look like phase-contrast velocity "
                            "data (see the checks above). Point the tool at "
                            "the PHASE series, or set "
                            "DICOM_STRICT_INPUT_CHECK = False to continue "
                            "anyway.")
                    print("  Continuing anyway "
                          "(DICOM_STRICT_INPUT_CHECK = False).\n")

                # ─ Build DICOM spatial grid in metres centred on isocenter ─
                pix_xy   = hdr.get("PixelSpacing_mm",
                                   [1000.0 * 2 * cfg.r_max_m / cfg.grid_N,
                                    1000.0 * 2 * cfg.r_max_m / cfg.grid_N])
                slice_mm = hdr.get("SliceThickness_mm", pix_xy[0])
                # Prefer the TRUE slice-to-slice step when the loader could
                # determine it (from SpacingBetweenSlices or the IPP span
                # projected onto the slice normal). SliceThickness is the
                # excitation width; with gaps or overlap the two differ, and
                # using the wrong one stretches/compresses every z position,
                # so Λ is evaluated at the wrong distance from isocentre.
                _sl_step = (hdr.get("SliceSpacing_mm")
                            if DICOM_GEOMETRIC_SLICE_SPACING else None)
                if _sl_step is not None and np.isfinite(_sl_step) \
                        and float(_sl_step) > 0:
                    if abs(float(_sl_step) - float(slice_mm)) \
                            > 0.01 * max(float(slice_mm), 1e-6):
                        print("  [DICOM] Slice step from geometry: "
                              f"{float(_sl_step):.3f} mm ≠ SliceThickness "
                              f"{float(slice_mm):.3f} mm — using the "
                              "geometric step (gapped/overlapping slices).")
                        print("           NOTE: this MOVES every z position "
                              "relative to a run made before this check "
                              "existed —")
                        print("           results for this dataset will "
                              "differ from such a run (the older z was "
                              "wrong).")
                    else:
                        print("  [DICOM] Slice step from geometry "
                              f"{float(_sl_step):.3f} mm agrees with "
                              f"SliceThickness {float(slice_mm):.3f} mm "
                              "— z positions unchanged.")
                    slice_mm = float(_sl_step)
                # DICOM PixelSpacing is [row_spacing, column_spacing]
                # (0028,0030). Row spacing is the distance between adjacent
                # ROWS, i.e. the step along the COLUMN-index axis Ny; column
                # spacing is the step along the row-index axis Nx. The loader
                # stores the pair in DICOM order, so pix_xy == [dy, dx] and
                # the indices below must be 1 for x and 0 for y.
                #
                # These were reversed until 2026-08. The error was invisible
                # on isotropic in-plane pixels (0.640625 x 0.640625 here) and
                # therefore changed no published number, but on an
                # anisotropic acquisition it transposed the exported VTK
                # spacing, the flow-rate area element and — worst — the
                # FALLBACK Lambda grid built below when the header carries no
                # ImagePositionPatient, which evaluated Lambda at transposed
                # coordinates. The IPP branch further down was always correct
                # (it reads col_sp = pix_xy[1], row_sp = pix_xy[0]), so the
                # two disagreed with each other.
                dx = float(pix_xy[1]) / 1000.0   # column spacing -> Nx  [m]
                dy = float(pix_xy[0]) / 1000.0   # row    spacing -> Ny  [m]
                dz = float(slice_mm)  / 1000.0
                Nxd, Nyd, Nzd = shape_d

                # ── Place the volume at its TRUE position relative to isocentre ──
                # Gradient nonlinearity is fixed in magnet coordinates (Λ=I at the
                # true isocentre). Using ImagePositionPatient + orientation puts each
                # voxel at its real distance from isocentre, instead of assuming the
                # volume is centred there. Falls back to centring if the geometry
                # is missing from the header.
                _ipp = hdr.get("ImagePositionPatient")
                _iop = hdr.get("ImageOrientationPatient")
                if (_ipp is not None and _iop is not None
                        and len(_ipp) == 3 and len(_iop) == 6):
                    row_dir   = np.asarray(_iop[0:3], float)   # +column index (Nx)
                    col_dir   = np.asarray(_iop[3:6], float)   # +row index    (Ny)
                    slice_dir = np.cross(row_dir, col_dir)     # slice normal  (Nz)
                    col_sp = float(pix_xy[1]); row_sp = float(pix_xy[0])
                    sl_sp  = float(slice_mm)                   # mm
                    I, J, K = np.meshgrid(np.arange(Nxd), np.arange(Nyd),
                                          np.arange(Nzd), indexing="ij")
                    Xd = (_ipp[0] + I*col_sp*row_dir[0] + J*row_sp*col_dir[0]
                          + K*sl_sp*slice_dir[0]) / 1000.0
                    Yd = (_ipp[1] + I*col_sp*row_dir[1] + J*row_sp*col_dir[1]
                          + K*sl_sp*slice_dir[1]) / 1000.0
                    Zd = (_ipp[2] + I*col_sp*row_dir[2] + J*row_sp*col_dir[2]
                          + K*sl_sp*slice_dir[2]) / 1000.0
                    _cx, _cy, _cz = Xd.mean()*1000, Yd.mean()*1000, Zd.mean()*1000
                    _roff = float(np.sqrt(_cx**2 + _cy**2 + _cz**2))
                    print("  [DICOM] Volume placed at TRUE position; centre "
                          f"({_cx:.0f}, {_cy:.0f}, {_cz:.0f}) mm, "
                          f"{_roff:.0f} mm from isocentre.")
                else:
                    xs_d = (np.arange(Nxd) - (Nxd - 1) / 2.0) * dx
                    ys_d = (np.arange(Nyd) - (Nyd - 1) / 2.0) * dy
                    zs_d = (np.arange(Nzd) - (Nzd - 1) / 2.0) * dz
                    Xd, Yd, Zd = np.meshgrid(xs_d, ys_d, zs_d, indexing="ij")
                    print("  [DICOM] No position in header; grid centred at "
                          "isocentre (assumed).")
                print(f"  [DICOM] Grid: {Nxd}×{Nyd}×{Nzd} voxels  "
                      f"({dx*1000:.2f}×{dy*1000:.2f}×{dz*1000:.2f} mm)  "
                      f"FOV ≈ {Nxd*dx*1000:.0f}×{Nyd*dy*1000:.0f}×"
                      f"{Nzd*dz*1000:.0f} mm")

                # ── Frame-of-reference guards (patient vs gradient frame) ──
                # NOTE — the assumption behind the block above, stated
                # explicitly. Λ lives in the GRADIENT-COIL frame: origin at
                # the magnet isocentre, axes along the physical gradient
                # axes. DICOM IPP/IOP live in the PATIENT-BASED (LPS)
                # frame (PS3.3 C.7.6.2), and nothing in the DICOM standard
                # pins that frame to the coil. The two coincide here by
                # VENDOR CONVENTION: on Siemens systems the LPS origin is
                # registered to isocentre and, for Head-First-Supine, the
                # LPS axes coincide with the physical gradient axes — the
                # same assumption made by gradunwarp on the same
                # coeff*.grad files, and by Lorenz et al. (MRM 2014), who
                # evaluate the identical field model on scanner-distortion-
                # corrected image coordinates. It can break when:
                #   · PatientPosition is not HFS (FFS flips x and z, prone
                #     flips x and y, ...): the axis identification fails;
                #   · the table moved between registration and scan: the
                #     origin shifts;
                #   · other vendors / newer exports: DICOM defines explicit
                #     equipment mappings (Image-to-Equipment Mapping
                #     Matrix, isocenter attributes) that should then be
                #     used instead.
                # The general fix is r_patient → r_equipment before
                # evaluating Λ; until that exists, non-HFS data must not
                # be corrected quantitatively through this path.
                _pp = str(DicomLoader._scanner_info.get(
                    "patient_position", "") or "").strip().upper()
                if _pp == "HFS":
                    print(f"  [DICOM] PatientPosition: {_pp} — LPS ↔ "
                          "gradient-axis identification valid (Siemens "
                          "convention).")
                elif _pp:
                    print("  " + "!" * 68)
                    print(f"  [DICOM] PatientPosition is '{_pp}', not HFS.")
                    print("          The LPS ↔ gradient-frame identification "
                          "assumed when evaluating Λ")
                    print("          holds for Head-First-Supine only; for "
                          f"'{_pp}' one or more axes are")
                    print("          flipped relative to the physical "
                          "gradient axes and the spatially")
                    print("          varying correction is evaluated at the "
                          "WRONG positions.")
                    print("          Do not use these corrected velocities "
                          "quantitatively.")
                    print("  " + "!" * 68)
                else:
                    print("  [DICOM] PatientPosition absent from header — "
                          "HFS assumed (see NOTE in code).")
                # ── Isocentre precondition: relative table position ──────
                # Λ's origin is the magnet isocentre. Reading the Siemens
                # private ImaRelTablePosition turns "the patient frame is
                # registered to isocentre" from an assumption into a
                # verified fact for this acquisition — the concrete form of
                # the "explicit isocentre handling" the frame-of-reference
                # NOTE above describes as future work for the general case.
                _tbl = locals().get("_tbl_rel", None)
                if _tbl is not None and len(_tbl) == 3:
                    _tbl_mag = float(np.linalg.norm(_tbl))
                    if _tbl_mag <= 1.0:
                        print("  [DICOM] Relative table position "
                              f"({_tbl[0]:.0f}, {_tbl[1]:.0f}, {_tbl[2]:.0f}) "
                              "mm — table at the reference position, so the")
                        print("          patient-frame origin coincides with "
                              "the magnet isocentre. Λ radii VERIFIED, not "
                              "assumed.")
                    else:
                        print("  " + "!" * 68)
                        print("  [DICOM] Relative table position is "
                              f"({_tbl[0]:.0f}, {_tbl[1]:.0f}, {_tbl[2]:.0f}) "
                              f"mm  (|.| = {_tbl_mag:.0f} mm), NOT zero.")
                        print("          The stored patient coordinates are "
                              "therefore offset from the magnet")
                        print("          isocentre by this vector, and Λ is "
                              "being evaluated at the WRONG radius.")
                        print("          Gradient nonlinearity grows with "
                              "radius, so the correction is")
                        print("          mis-scaled across the whole volume. "
                              "Subtract this offset from the")
                        print("          coordinates before trusting "
                              "quantitative results.")
                        print("  " + "!" * 68)
                else:
                    print("  [DICOM] Relative table position not in the header "
                          "— isocentre offset UNVERIFIED (assumed zero).")

                if _iop is not None and len(_iop) == 6:
                    _rd = np.asarray(_iop[0:3], float)
                    _cd = np.asarray(_iop[3:6], float)
                    _nd = np.cross(_rd, _cd)
                    def _ang_to_axis(v):
                        v = v / max(np.linalg.norm(v), 1e-12)
                        return float(np.degrees(
                            np.arccos(min(1.0, np.max(np.abs(v))))))
                    _obl = max(_ang_to_axis(_rd), _ang_to_axis(_cd),
                               _ang_to_axis(_nd))
                    print(f"  [DICOM] Stack obliquity: worst axis-alignment "
                          f"angle {_obl:.2f}° (row/col/normal vs LPS axes).")
                    if _obl > 5.0:
                        print("  " + "!" * 68)
                        print(f"  [DICOM] OBLIQUE ACQUISITION ({_obl:.1f}° "
                              "> 5°).")
                        print("          Two approximations degrade with "
                              "obliquity:")
                        print("          1. encoding directions are snapped "
                              "to the nearest axis (argmax),")
                        print("             so each velocity component mixes "
                              "neighbouring axes;")
                        print("          2. Λ is applied along the LPS axes, "
                              "not the encoding axes.")
                        print("          The general fix is the full "
                              "rotation v_grad = R·v_patient before")
                        print("          correction (and R⁻¹ after); until "
                              "implemented, restrict")
                        print("          quantitative use to near-axis-"
                              "aligned stacks.")
                        print("  " + "!" * 68)

                # ─ Recompute Λ on the DICOM grid ─
                t0 = time.perf_counter()
                Lambda_d = evaluator.compute_lambda(Xd, Yd, Zd, eps=cfg.eps_m)
                inside_r0_d = evaluator.inside_r0_mask(
                    Xd, Yd, Zd, margin=cfg.eps_m)
                # True geometric r<=R0 fraction, captured BEFORE the optional
                # magnitude signal mask is AND-ed into inside_r0_d below. Without
                # this, later reports would print the signal fraction under an
                # "inside R0" label (e.g. 62%) and contradict the 100% line here.
                r0_only_frac     = float(inside_r0_d.mean())   # fraction inside R0 (geometry only)
                signal_only_frac = None                        # set below iff a magnitude mask is applied
                print("  [DICOM] Λ recomputed on DICOM grid: "
                      f"{time.perf_counter()-t0:.1f} s; "
                      f"{int(inside_r0_d.sum())}/{inside_r0_d.size} voxels "
                      f"inside r ≤ R0 ({100*inside_r0_d.mean():.1f}%).")

                # Optional magnitude mask: PC-MRI background is random noise
                # spanning ±VENC, which dominates the stats. If a magnitude
                # series is provided, restrict the analysis to signal voxels.
                mag_for_plot = None
                sigma_v_map  = None       # per-voxel velocity noise [m/s]
                _mag_pre = hdr.get("magnitude_volume")   # from multi-folder loader
                if _mag_pre is not None or DICOM_MAGNITUDE_DIR:
                    mag_vol = (_mag_pre if _mag_pre is not None
                               else DicomLoader.load_magnitude(DICOM_MAGNITUDE_DIR))
                    if mag_vol is not None and mag_vol.shape == inside_r0_d.shape:
                        _mask_record = None
                        if DICOM_MAGNITUDE_MASK_FRAC is None:
                            _nz0 = mag_vol[mag_vol > 0]
                            _t0 = _otsu_threshold(_nz0) if _nz0.size else _otsu_threshold(mag_vol)
                            _bg0 = float(np.median(_nz0[_nz0 <= _t0])) if _nz0.size else 0.0
                            DICOM_MAGNITUDE_MASK_FRAC = float(
                                0.5 * (_bg0 + _t0) / max(float(mag_vol.max()), 1e-12))
                            print("  [DICOM] Signal-mask threshold chosen from the data: "
                                  f"{DICOM_MAGNITUDE_MASK_FRAC * float(mag_vol.max()):.0f} "
                                  f"({100*DICOM_MAGNITUDE_MASK_FRAC:.0f}% of the maximum),")
                            print(f"          halfway between the background level ({_bg0:.0f}) and the "
                                  f"object/background split (Otsu, {_t0:.0f}). Override: --mask-frac F.")
                            _mask_record = {
                                "source": "automatic: halfway between background level and Otsu split",
                                "fraction": DICOM_MAGNITUDE_MASK_FRAC,
                                "threshold": DICOM_MAGNITUDE_MASK_FRAC * float(mag_vol.max()),
                                "background_level": _bg0, "otsu": float(_t0)}
                        elif DICOM_MASK_FRAC_EXPLICIT:
                            _mask_record = {
                                "source": "command line (--mask-frac)",
                                "fraction": float(DICOM_MAGNITUDE_MASK_FRAC),
                                "threshold": float(DICOM_MAGNITUDE_MASK_FRAC) * float(mag_vol.max())}
                        sig = mag_vol > (DICOM_MAGNITUDE_MASK_FRAC
                                         * float(mag_vol.max()))
                        # A fraction OF THE MAXIMUM only separates object from
                        # air when the magnitude image uses its dynamic range.
                        # Some acquisitions do not: if the peak sits at, say,
                        # 139 of 4095 possible levels, then 10% of it is 14 —
                        # below the air noise, so nearly every voxel passes and
                        # the "signal mask" masks nothing. Everything
                        # downstream then measures air: mean |v| comes out near
                        # VENC and the strict input check rejects perfectly
                        # good phase data.
                        # Detect that by how much was kept, and fall back to
                        # Otsu, which finds the threshold that best separates
                        # the two populations regardless of their absolute
                        # level.
                        _kept = float(sig.mean())
                        if _kept > 0.80:
                            _t_otsu = _otsu_threshold(mag_vol)
                            _sig2 = mag_vol > _t_otsu
                            print(f"  [DICOM] Magnitude mask kept "
                                  f"{100*_kept:.0f}% of the volume — the "
                                  f"{DICOM_MAGNITUDE_MASK_FRAC:.0%}-of-max "
                                  f"threshold ({DICOM_MAGNITUDE_MASK_FRAC * float(mag_vol.max()):.0f}) "
                                  f"is below the\n          air noise "
                                  f"(magnitude peaks at only "
                                  f"{float(mag_vol.max()):.0f}). Falling back "
                                  f"to Otsu: threshold {_t_otsu:.0f}, keeps "
                                  f"{100*float(_sig2.mean()):.0f}%.")
                            if 0.02 < float(_sig2.mean()) < 0.80:
                                sig = _sig2
                            else:
                                print("  [DICOM] Otsu did not separate them "
                                      "either — check the magnitude series.")
                        # ── automatic recovery from a mask that admits background ──
                        # (see DICOM_AUTO_MASK_RECOVERY). Runs BEFORE the mask is
                        # applied, so everything downstream sees the recovered one.
                        _mask_recovery = None
                        if DICOM_AUTO_MASK_RECOVERY and not DICOM_MASK_FRAC_EXPLICIT:
                            try:
                                _venc_pre = float(hdr.get("VENC", cfg.VENC))
                                _ins = inside_r0_d & sig
                                _nt_pre = v_dcm.shape[-1] if v_dcm.ndim == 5 else 1
                                _acc = np.zeros(int(_ins.sum()), np.float64)
                                for _tt in range(_nt_pre):
                                    _vt = v_dcm[..., _tt] if v_dcm.ndim == 5 else v_dcm
                                    _acc += np.sqrt((_vt[:, _ins] ** 2).sum(axis=0))
                                _spd_pre = np.zeros(_ins.shape, np.float32)
                                _spd_pre[_ins] = _acc / _nt_pre
                                del _acc
                                _dg0 = _mask_background_diagnosis(
                                    mag_vol, _ins, _spd_pre, _venc_pre,
                                    DICOM_MAGNITUDE_MASK_FRAC)
                                if _dg0 is not None and _dg0["background_in_mask"]:
                                    _nzv = mag_vol[mag_vol > 0]
                                    _t_o = _dg0["otsu"]
                                    _bg_lvl = float(np.median(_nzv[_nzv <= _t_o]))
                                    _cands = [0.5 * (_bg_lvl + _t_o), _t_o]
                                    for _t_rec in _cands:
                                        if _t_rec <= _dg0["used_threshold"]:
                                            continue
                                        _sig_rec = mag_vol > _t_rec
                                        # the new mask is a subset of the old one, so the
                                        # speeds computed above still apply. Accept it when
                                        # the share of masked voxels above VENC is within the
                                        # 2% the magnitude check itself tolerates; an EMPTY
                                        # background group is success, not "cannot tell".
                                        _ins1 = inside_r0_d & _sig_rec
                                        if int(_ins1.sum()) < 50:
                                            continue
                                        _over1 = float((_spd_pre[_ins1] > _venc_pre).mean())
                                        if _over1 <= 0.02:
                                            _mmax = max(float(mag_vol.max()), 1e-12)
                                            print()
                                            print("  " + "=" * 68)
                                            print("  [DICOM] MASK AUTO-RECOVERED — the signal mask admitted "
                                                  "background (random phase).")
                                            print(f"          preset '{DATASET_PRESET}': {100*_dg0['frac_used']:.0f}% of the "
                                                  f"maximum (threshold {_dg0['used_threshold']:.0f}) kept "
                                                  f"{_dg0['kept_pct_of_volume']:.0f}% of the volume;")
                                            print(f"          {100*_dg0['over_all']:.0f}% of the masked voxels exceeded VENC, "
                                                  "all of them background.")
                                            print(f"          Threshold raised to {_t_rec:.0f} ({100*_t_rec/_mmax:.0f}% of the "
                                                  f"maximum): halfway between the background level ({_bg_lvl:.0f})")
                                            print(f"          and the object/background split (Otsu, {_t_o:.0f}). "
                                                  f"Now keeps {100*float(_ins1.mean()):.1f}% of the volume; "
                                                  f"{100*_over1:.1f}% exceed VENC.")
                                            print("          AUTOMATIC, NOT TUNED: confirm with a threshold sweep and "
                                                  "re-run with --mask-frac F")
                                            print("          for the numbers you report (above the background, "
                                                  "BELOW the dimmest part of the object).")
                                            print("  " + "=" * 68)
                                            print()
                                            _mask_recovery = {
                                                "original_fraction": _dg0["frac_used"],
                                                "original_threshold": _dg0["used_threshold"],
                                                "recovered_threshold": float(_t_rec),
                                                "recovered_fraction": float(_t_rec / _mmax),
                                                "background_level": _bg_lvl,
                                                "otsu": float(_t_o),
                                                "kept_pct_before": _dg0["kept_pct_of_volume"],
                                                "kept_pct_after": 100.0 * float(_ins1.mean()),
                                                "note": "automatic, not tuned; confirm with a sweep",
                                            }
                                            sig = _sig_rec
                                            break
                                del _spd_pre
                            except Exception as _mre:
                                print(f"  [DICOM] mask recovery skipped ({type(_mre).__name__}: {_mre})")
                        n_before = int(inside_r0_d.sum())
                        inside_r0_d = inside_r0_d & sig
                        # Fraction of the whole grid that is inside R0 AND above
                        # the signal threshold — this is the analysis set, and is
                        # the voxels genuinely inside R0.
                        # Keep the ORIGINAL air-only background. The noise
                        # level sigma_M must be measured against AIR. If the
                        # brain mask is allowed to narrow `sig` first, then
                        # ~sig includes the bright SCALP, std() jumps ~5x, SNR
                        # collapses and every derived threshold inflates past
                        # VENC — which is physically impossible.
                        sig_air = sig.copy()
                        if DICOM_BRAIN_MASK:
                            _brain = _build_brain_mask(
                                mag_vol, DICOM_MAGNITUDE_MASK_FRAC,
                                DICOM_BRAIN_MASK_ERODE)
                            if _brain is not None:
                                _n_pre = int(inside_r0_d.sum())
                                inside_r0_d = inside_r0_d & _brain
                                sig = sig & _brain
                                print("  [DICOM] Brain mask applied: "
                                      f"{int(inside_r0_d.sum()):,} voxels "
                                      f"(was {_n_pre:,}).")
                        # ── pipe rig: analysis set = the pipe only ────────────────────
                        # (see DICOM_PIPE_MAIN_OBJECT_ONLY). After sig_air was taken.
                        _pipe_extra = None
                        if str(DATASET_PRESET) == "pipe_flow" and DICOM_PIPE_MAIN_OBJECT_ONLY:
                            try:
                                from scipy import ndimage as _ndi_p
                                _labp, _nobj = _ndi_p.label(inside_r0_d)
                                if _nobj > 1:
                                    _szp = _ndi_p.sum(inside_r0_d, _labp, index=range(1, _nobj + 1))
                                    _keep = _labp == (int(np.argmax(_szp)) + 1)
                                    _out = inside_r0_d & ~_keep
                                    _n_out, _n_keep = int(_out.sum()), int(_keep.sum())
                                    _vv = v_dcm[:, _out]
                                    _sx = np.sqrt((_vv ** 2).sum(axis=0))
                                    _sx = _sx.mean(axis=-1) if _sx.ndim > 1 else _sx
                                    _vnc = float(hdr.get("VENC", cfg.VENC))
                                    _med, _fov = float(np.median(_sx)), float((_sx > _vnc).mean())
                                    del _vv, _sx
                                    inside_r0_d = _keep
                                    sig = sig & _keep
                                    print("  [DICOM] Pipe preset: analysis set restricted to the main object "
                                          f"({_n_keep:,} voxels).")
                                    print(f"          Excluded {_n_out:,} voxels in {_nobj - 1} smaller object(s): "
                                          f"median |v| {100*_med:.1f} cm/s, {100*_fov:.0f}% above VENC")
                                    # only call it random phase when the numbers say so
                                    # (uniform phase: median |v| ~ 0.95 VENC, ~48% above VENC)
                                    if _med >= 0.8 * _vnc and _fov >= 0.30:
                                        print("          Random phase, like empty background: no usable velocity.")
                                    print("          Keep them: DICOM_PIPE_MAIN_OBJECT_ONLY = False.")
                                    _pipe_extra = {"main_object_voxels": _n_keep,
                                                   "excluded_voxels": _n_out,
                                                   "excluded_objects": int(_nobj - 1),
                                                   "excluded_median_speed_cm_s": 100 * _med,
                                                   "excluded_fraction_above_venc": _fov}
                            except Exception as _pex:
                                print(f"  [DICOM] main-object restriction skipped "
                                      f"({type(_pex).__name__}: {_pex})")
                        signal_only_frac = float(inside_r0_d.mean())
                        mag_for_plot = mag_vol
                        print("  [DICOM] Magnitude signal mask applied: "
                              f"{int(inside_r0_d.sum()):,} signal voxels "
                              f"(was {n_before:,}; background noise excluded).")

                        _mask_banner_done = False
                        # ── phase-assembly (layout) check ───────────────
                        # Only meaningful INSIDE the object. Over the whole
                        # volume the number is dominated by air, where each
                        # component is uniform on [-VENC,+VENC] and |v|
                        # exceeds VENC for 1 - pi/6 = 47.6% of voxels purely
                        # by geometry — so an unmasked test reports 25-45% on
                        # any phantom regardless of correctness. Masked, a
                        # correctly assembled volume gives ~0% and a
                        # BLOCKED/INTERLEAVED mix-up gives tens of percent,
                        # because each voxel then combines components of
                        # three DIFFERENT physical voxels.
                        try:
                            # VENC_d is not assigned until much later in this
                            # function, so read it the same way that later
                            # assignment does. Referencing VENC_d here raised
                            # UnboundLocalError on every run.
                            _venc = float(hdr.get("VENC", cfg.VENC))
                            _sp = np.linalg.norm(v_dcm, axis=0)   # (X,Y,Z,T)
                            _in = _sp[inside_r0_d]
                            _in = _in[np.isfinite(_in)]
                            if _in.size:
                                _fo = float((_in > _venc).mean())
                                _dg = None
                                if _fo > 0.10:
                                    _dg = _mask_background_diagnosis(
                                        mag_vol, inside_r0_d, _sp.mean(axis=-1),
                                        _venc, DICOM_MAGNITUDE_MASK_FRAC)
                                if _fo > 0.10 and _dg is not None \
                                        and _dg["background_in_mask"]:
                                    _print_mask_background_banner(_dg, DATASET_PRESET)
                                    _mask_banner_done = True
                                elif _fo > 0.10:
                                    print()
                                    print("  " + "#" * 68)
                                    print(f"  [DICOM] LAYOUT WARNING: "
                                          f"{100*_fo:.0f}% of voxels INSIDE "
                                          "the object exceed VENC.")
                                    print("          Correct phase assembly "
                                          "gives ~0% here, so the")
                                    print("          frame→(direction,slice) "
                                          "mapping is probably wrong —")
                                    print("          a BLOCKED vs INTERLEAVED "
                                          "storage-order mismatch")
                                    print("          ('all x then all y then "
                                          "all z' vs 'x,y,z per")
                                    print("          time point'). Each voxel "
                                          "would then mix components")
                                    print("          from three different "
                                          "physical voxels.")
                                    print("          If per-direction folders "
                                          "exist, load those instead.")
                                    print("          DICOM_DIRECTION_ORDER "
                                          "cannot fix this — it only")
                                    print("          relabels axes.")
                                    print("  " + "#" * 68)
                                    print()
                                else:
                                    print("  [DICOM] Layout check OK: "
                                          f"{100*_fo:.1f}% of masked voxels "
                                          "exceed VENC (valid assembly).")
                                # dicom_summary is not built until much later
                                # in this function (it is still None here), so
                                # stash the value on a local and merge it in
                                # when the dict is created. Assigning into the
                                # None directly raised TypeError, which the
                                # except below swallowed as "check skipped".
                                _layout_frac_over_venc = _fo
                        except Exception as _le:
                            print(f"  [DICOM] Layout check skipped "
                                  f"({type(_le).__name__}: {_le})")

                        # ── magnitude-data check ────────────────────────
                        # Now that background is excluded this becomes a real
                        # discriminator: inside the object, genuine flow is a
                        # few percent of VENC, whereas a magnitude series read
                        # as phase saturates near VENC everywhere.
                        try:
                            _spd = np.sqrt((v_dcm ** 2).sum(axis=0))  # |v| vol
                            _sp_sig = float(_spd.mean(axis=-1)[
                                    inside_r0_d].mean())
                            _venc_h2 = float(hdr.get("VENC", cfg.VENC))
                            _pc_sig = 100 * _sp_sig / max(_venc_h2, 1e-9)
                            print("  [DICOM] Mean |v| within the mask: "
                                  f"{_sp_sig*100:.1f} cm/s "
                                  f"({_pc_sig:.0f}% of VENC)")

                            # Distinguish TWO cases that both give a high mean:
                            #   (A) magnitude data misread as phase — the whole
                            #       object saturates, so a LARGE fraction of
                            #       voxels sit ABOVE VENC (impossible for real
                            #       phase) and there is little stationary signal.
                            #   (B) a genuine high-flow phantom (e.g. a pipe at
                            #       a high pump rate filling most of the FOV) —
                            #       the mean is high, but NOTHING exceeds VENC
                            #       and there is a real spread of velocities.
                            # The mean alone cannot tell these apart; the
                            # fraction of voxels ABOVE VENC can. Magnitude-as-
                            # phase pushes many voxels past VENC; genuine flow
                            # never exceeds it. Use that as the discriminator.
                            _in_mask = _spd.mean(axis=-1)[inside_r0_d]
                            _frac_over = float(
                                (_in_mask > _venc_h2).mean())
                            _looks_like_magnitude = (
                                _pc_sig > 60.0 and _frac_over > 0.02)

                            if _pc_sig > 60.0:
                                print(f"  [DICOM] {100*_frac_over:.1f}% of masked "
                                      f"voxels exceed VENC "
                                      f"({'>2% → magnitude-like' if _frac_over > 0.02 else '≤2% → consistent with genuine fast flow'}).")

                            _dg2 = None
                            if _looks_like_magnitude:
                                _dg2 = _mask_background_diagnosis(
                                    mag_vol, inside_r0_d, _spd.mean(axis=-1),
                                    _venc_h2, DICOM_MAGNITUDE_MASK_FRAC)
                            if _looks_like_magnitude and _dg2 is not None \
                                    and _dg2["background_in_mask"]:
                                # Same alarm, correct cause: the mask holds
                                # background. Stop (an analysis of air is
                                # meaningless) but say what is actually wrong.
                                if not _mask_banner_done:
                                    _print_mask_background_banner(_dg2, DATASET_PRESET)
                                if DICOM_STRICT_INPUT_CHECK:
                                    raise ValueError(
                                        "The signal mask includes background: "
                                        f"{_dg2['kept_pct_of_volume']:.0f}% of the "
                                        "volume is kept at "
                                        f"{100*_dg2['frac_used']:.0f}% of the magnitude "
                                        "maximum, and the excess above VENC is in the "
                                        "low-magnitude voxels only. The phase data are "
                                        "fine. Raise the mask fraction (try "
                                        f"{_dg2['suggested_frac']:.2f}) with --mask-frac.")
                                print("  Continuing anyway "
                                      "(DICOM_STRICT_INPUT_CHECK = False).\n")
                            elif _looks_like_magnitude:
                                print()
                                print("  " + "!" * 68)
                                print("  [DICOM] THIS DOES NOT LOOK LIKE PHASE "
                                      "DATA: mean |v| inside the")
                                print(f"          object is {_pc_sig:.0f}% of "
                                      "VENC AND "
                                      f"{100*_frac_over:.0f}% of voxels exceed")
                                print("          VENC, which is impossible for "
                                      "real phase. A MAGNITUDE")
                                print("          series read as phase saturates "
                                      "this way. Check you")
                                print("          selected the PHASE series "
                                      "(~3x as many files as the")
                                print("          magnitude series beside it).")
                                print("          If it IS the phase series, "
                                      "the phase->velocity zero point is")
                                print("          wrong instead: re-read the "
                                      "[DICOM] axis-x/y/z lines above and")
                                print("          check that the reported "
                                      "range is centred on 0 cm/s.")
                                print("  " + "!" * 68)
                                if DICOM_STRICT_INPUT_CHECK:
                                    raise ValueError(
                                        "Mean |v| inside the mask is "
                                        f"{_pc_sig:.0f}% of VENC and "
                                        f"{100*_frac_over:.0f}% of voxels exceed "
                                        "VENC — this looks like magnitude data, "
                                        "not phase. Set DICOM_STRICT_INPUT_CHECK "
                                        "= False to continue anyway.")
                                print("  Continuing anyway "
                                      "(DICOM_STRICT_INPUT_CHECK = False).\n")
                            elif _pc_sig > 60.0:
                                # High mean but nothing over VENC → accept it as
                                # a genuine high-flow acquisition, with a note.
                                print("  [DICOM] High mean |v| but no voxels "
                                      "exceed VENC and the phase zero-point is "
                                      "consistent → treating this as GENUINE "
                                      "fast flow (e.g. a high-rate flow "
                                      "phantom), not magnitude data. Proceeding.")
                        except ValueError:
                            raise
                        except Exception:
                            pass

                        # ── Per-voxel velocity noise from local SNR ─────────
                        # Velocity noise is NOT uniform across a head:
                        #     σ_v(r) = √2·VENC / (π·SNR(r))     [see add_phase_noise]
                        # Brain parenchyma runs SNR≈10 (σ_v≈4 cm/s), but scalp,
                        # orbit and skull-base voxels can be SNR≈2-3, where
                        # σ_v reaches 14-20 cm/s. A single global threshold
                        # therefore either keeps scalp noise or throws away real
                        # vessels. Estimating σ_v per voxel fixes both.
                        try:
                            _bg = mag_vol[~sig_air]     # AIR only, never scalp
                            _sigma_M = float(np.std(_bg)) if _bg.size > 100 else None
                            if _sigma_M and _sigma_M > 0:
                                _snr_map = mag_vol / _sigma_M
                                sigma_v_map = (np.sqrt(2.0) * float(hdr.get("VENC", cfg.VENC)) /
                                               (np.pi * np.maximum(_snr_map, 1e-3)))
                                _venc_h = float(hdr.get("VENC", cfg.VENC))
                                # Nothing can be measured faster than VENC, so
                                # a noise estimate implying that is wrong by
                                # construction. Cap it and say so loudly rather
                                # than silently emptying the figure.
                                _lim = DICOM_MAX_SIGMA_V_FRAC * _venc_h
                                _unusable = sigma_v_map > _lim
                                _n_hi = int((_unusable & inside_r0_d).sum())
                                if _n_hi:
                                    _n_pre_u = int(inside_r0_d.sum())
                                    inside_r0_d = inside_r0_d & (~_unusable)
                                    print(f"  [DICOM] Excluded {_n_hi:,} voxels with "
                                          f"σ_v > {100*DICOM_MAX_SIGMA_V_FRAC:.0f}% of "
                                          f"VENC ({_lim*100:.0f} cm/s) — signal "
                                          "dropout, no usable velocity "
                                          f"({_n_pre_u:,} -> {int(inside_r0_d.sum()):,}).")
                                    signal_only_frac = float(inside_r0_d.mean())
                                _sv = sigma_v_map[inside_r0_d]
                                print("  [DICOM] Velocity-noise map from local SNR: "
                                      f"σ_v median {np.median(_sv)*100:.1f} cm/s, "
                                      f"p90 {np.percentile(_sv,90)*100:.1f} cm/s "
                                      f"(background σ_M = {_sigma_M:.1f}, "
                                      "median SNR ≈ "
                                      f"{np.sqrt(2)*_venc_h/(np.pi*max(float(np.median(_sv)),1e-9)):.0f})")
                        except Exception as _exc:
                            print(f"  [DICOM] SNR map unavailable ({_exc}); "
                                  "falling back to a global noise estimate.")

                        # ── Which component carries the flow? ──────────────
                        # The one direction check that needs no header tag: in
                        # a phantom the flow direction is known, so the
                        # dominant component confirms or refutes the vx/vy/vz
                        # assignment.
                        #
                        # This must run AFTER the magnitude mask. Background
                        # phase is uniform over +/-VENC, so its components have
                        # a standard deviation of VENC/sqrt(3) ~ 87 cm/s on
                        # this dataset — far larger than the real flow of
                        # ~14 cm/s. Any selection made before masking picks
                        # background, where the three components are equal by
                        # symmetry, and reports a meaningless 33/33/33.
                        try:
                            _vm = v_dcm.mean(axis=-1)          # (3,Nx,Ny,Nz)
                            _cm = np.sqrt((_vm ** 2).sum(axis=0))
                            _sel = inside_r0_d & (
                                _cm > np.percentile(_cm[inside_r0_d], 90))
                            if int(_sel.sum()) > 50:
                                _fr = [float(np.mean(np.abs(_vm[a][_sel])))
                                       for a in range(3)]
                                _tt = sum(_fr) or 1.0
                                _pc = [100 * f / _tt for f in _fr]
                                _dom = int(np.argmax(_pc))
                                print("  [DICOM] Flow direction (top 10% most "
                                      "time-coherent SIGNAL voxels, "
                                      f"n={int(_sel.sum()):,}):")
                                print(f"           vx {_pc[0]:.0f}%   "
                                      f"vy {_pc[1]:.0f}%   vz {_pc[2]:.0f}%   "
                                      f"→ dominant: v{'xyz'[_dom]}")
                                if max(_pc) < 45:
                                    print("           No single component "
                                          "dominates — the flow is oblique, or "
                                          "the phantom has")
                                    print("           several compartments "
                                          "flowing in different directions.")
                                else:
                                    print("           Compare with the known "
                                          "flow direction of your phantom. If "
                                          "they disagree,")
                                    print("           set DICOM_DIRECTION_ORDER "
                                          "to permute the components.")
                        except Exception as _exc:
                            print("  [DICOM] Flow-direction summary "
                                  f"unavailable ({_exc}).")

                        # ── pipe rig: put the FLOW frame on the tube axis ──────────────
                        # (see DICOM_AUTO_AXIS_ASSIGN). Runs before the correction,
                        # so Lambda's rows are applied to the right components.
                        _axis_autoassign = None
                        if (DICOM_AUTO_AXIS_ASSIGN and str(DATASET_PRESET) == "pipe_flow"
                                and hdr.get("direction_source") == "assumed"
                                and v_dcm.shape[0] == 3):
                            try:
                                _aa = _pipe_axis_assignment(
                                    v_dcm, inside_r0_d, locals().get("_iop", None),
                                    (dx, dy, dz))
                                if _aa is not None and _aa["mismatch"]:
                                    _g, _f = _aa["tube_axis"], _aa["flow_component"]
                                    _tmp = v_dcm[_g].copy()
                                    v_dcm[_g] = v_dcm[_f]
                                    v_dcm[_f] = _tmp
                                    del _tmp
                                    print()
                                    print("  " + "=" * 68)
                                    print("  [DICOM] AXES AUTO-ASSIGNED (pipe preset) — the header gives no "
                                          "velocity-encoding order.")
                                    print(f"          Tube: straight along {'xyz'[_g]} "
                                          f"({_aa['extent_mm'][_g]:.0f} mm long, elongation "
                                          f"{_aa['elongation']:.1f}x). Flow: "
                                          f"{100*_aa['shares'][_f]:.0f}% in component {'xyz'[_f]}, "
                                          f"{100*_aa['shares'][_g]:.0f}% along the tube.")
                                    print(f"          Flow runs along a tube, so components {'xyz'[_g]} and "
                                          f"{'xyz'[_f]} were swapped: the flow is now {'xyz'[_g]}.")
                                    print("          Only the FLOW frame is determined by this. The other two "
                                          "frames keep their acquisition order")
                                    print("          (their order cannot be told from the data), and the SIGN of "
                                          "the flow is not determined.")
                                    print("          To set the order yourself:  --direction-order XYZ")
                                    print("  " + "=" * 68)
                                    print()
                                    _axis_autoassign = {
                                        "tube_axis": "xyz"[_g], "flow_component_before": "xyz"[_f],
                                        "swapped": f"{'xyz'[_g]}<->{'xyz'[_f]}",
                                        "elongation": _aa["elongation"],
                                        "extent_mm": _aa["extent_mm"], "shares_before": _aa["shares"],
                                        "note": "flow frame determined from tube geometry; other frames "
                                                "and the flow sign are not"}
                            except Exception as _aae:
                                print(f"  [DICOM] axis auto-assignment skipped "
                                      f"({type(_aae).__name__}: {_aae})")

                        # ── Anatomical sign verification (intracranial) ──
                        # ONLY on brain data. The six tests are textbook facts
                        # about the circle of Willis — left/right MCA, ICA
                        # terminus, basilar, ACA, PCA — and they are meaningless
                        # anywhere else. On the ideal phantom this fired and
                        # reported "SIGNS DISAGREE ... anatomy is consistent
                        # with (-1, 1, 1)": four of the six ROIs correctly found
                        # zero voxels, but the PCA box caught 682 voxels of
                        # plastic and water and declared an inversion from them.
                        # Acting on that would have corrupted the phantom run on
                        # the strength of a brain prior applied to a pipe rig.
                        # A prior is only evidence where the prior applies.
                        _is_brain = (str(DATASET_PRESET).lower()
                                     in ("in_vivo_head", "manual"))
                        if DICOM_VERIFY_SIGNS_ANATOMICALLY and not _is_brain:
                            print("  [DICOM] Anatomical sign check SKIPPED: it "
                                  "encodes circle-of-Willis anatomy and the "
                                  f"preset is '{DATASET_PRESET}'.")
                            print("          On a phantom the flow direction is "
                                  "known from the rig instead — check that the "
                                  "dominant")
                            print("          velocity component matches the tube "
                                  "axis reported by the flux-conservation block.")
                        if DICOM_VERIFY_SIGNS_ANATOMICALLY and _is_brain:
                            try:
                                _sgn_report = _verify_signs_anatomically(
                                    v_dcm, Xd * 1000.0, Yd * 1000.0,
                                    Zd * 1000.0, inside_r0_d,
                                    mag_for_plot)
                                if _sgn_report is not None:
                                    vessel_stats_sign = _sgn_report
                            except Exception as _sx:
                                print("  [DICOM] anatomical sign check "
                                      f"skipped ({_sx}).")
                    elif mag_vol is not None:
                        _src = (Path(DICOM_MAGNITUDE_DIR).name if DICOM_MAGNITUDE_DIR
                                else "(multi-folder magnitude)")
                        print(f"  [DICOM] Magnitude '{_src}' grid "
                              f"{'×'.join(map(str, mag_vol.shape))} ≠ velocity grid "
                              f"{'×'.join(map(str, inside_r0_d.shape))}; signal mask skipped.")

                # ─ Correction pipeline for reconstructed DICOM ─
                # VENC from the DICOM header — needed downstream by fig05
                # and run_config.
                VENC_d   = float(hdr.get("VENC", cfg.VENC))

                # ── fig03 rendered with the ACQUIRED VENC ────────────────
                # Stage 4 rendered the spatially-varying-VENC figure with the
                # SIMULATION VENC (cfg.VENC). If the acquired series uses a
                # different value, the cm/s maps and the title are wrong for
                # this dataset (the % histogram is VENC-independent), so
                # re-render it now with the detected value — it overwrites
                # the earlier file.
                if abs(VENC_d - float(cfg.VENC)) > 1e-3:
                    print(f"  [DICOM] Acquired VENC {VENC_d*100:.0f} cm/s ≠ "
                          f"simulation VENC {cfg.VENC*100:.0f} cm/s — "
                          f"re-rendering {_fig_label(17).strip('[]')} with "
                          "the acquired value.")
                    try:
                        plotter.plot_spatially_varying_venc(
                            Lambda, VENC_d, xv_mm, iy,
                            inside_r0=inside_r0, y_mm=_y_slice_mm)
                    except Exception as _e17:
                        print("  [DICOM] fig re-render skipped "
                              f"({type(_e17).__name__}: {_e17})")
                    print("  [DICOM] The simulation figures were drawn before "
                          "this series was read, so they used")
                    print(f"          VENC = {cfg.VENC:.2f} m/s and Nt = "
                          f"{cfg.Nt}. Re-run with --sim-venc {VENC_d:.2f} "
                          f"--sim-nt {Nt_d}")
                    print("          to make fig02/fig04 describe the same "
                          "measurement conditions as this data.")

                print("  [DICOM] Stage 3: Gradient nonlinearity correction "
                      "Λ(r)⁻¹ …")
                t0 = time.perf_counter()
                v_corr_d = corrector.correct_4d(v_dcm, Lambda_d)
                # Λ-deviation field ‖Λ−I‖_F per voxel, for the fig05
                # quantitative validation (correlate with the |Δv| map).
                _I3 = np.eye(3)[:, :, None, None, None]
                lambda_dev_d = np.sqrt(
                    np.sum((Lambda_d - _I3) ** 2, axis=(0, 1)))   # (Nx,Ny,Nz)
                # Lambda_d is a (3,3,Nx,Ny,Nz) float64 field — 303 MiB on a
                # 224x224x88 grid — and this is its last consumer. Releasing
                # it here keeps that memory out of the statistics block below.
                del Lambda_d, _I3
                print("           Correction time: "
                      f"{time.perf_counter()-t0:.2f} s")

                # ─ Correction magnitude statistics (no ground truth) ─────
                #
                # MEMORY — this block used to be the peak of the whole run.
                #
                # It built a full (3,Nx,Ny,Nz,Nt) difference array and three
                # (Nx,Ny,Nz,Nt) magnitude volumes, and np.linalg.norm
                # allocates a temporary the size of its INPUT before reducing
                # over the axis. For 3x224x224x88x11 float32 that is
                #   v_dcm 556 + v_corr_d 556 + delta 556 + norm temp 556
                #   + 3 x 185 (magnitudes) + Lambda_d 303  =  ~3.0 GiB
                # and it died with
                #   MemoryError: Unable to allocate 185 MiB for an array with
                #   shape (224, 224, 88, 11) and data type float32
                # AFTER a 680 s Lambda computation and a completed correction
                # — the expensive work was thrown away for want of the last
                # 185 MiB.
                #
                # Nothing downstream uses the 4-D magnitudes: only the three
                # TIME AVERAGES below are read. So accumulate them one cardiac
                # phase at a time and never materialise the 4-D arrays or the
                # difference at all. Peak drops from ~3.0 to ~1.15 GiB, and
                # the extra working set is one 3-D frame (~17 MiB).
                _shape3 = tuple(v_dcm.shape[1:4])
                _Ntd    = int(v_dcm.shape[-1])
                mag_meas_avg  = np.zeros(_shape3, dtype=np.float32)
                mag_corr_avg  = np.zeros(_shape3, dtype=np.float32)
                mag_delta_avg = np.zeros(_shape3, dtype=np.float32)
                _acc = np.empty(_shape3, dtype=np.float32)
                _tmp = np.empty(_shape3, dtype=np.float32)
                for _t in range(_Ntd):
                    # |v| for measured and corrected
                    for _src, _dst in ((v_dcm, mag_meas_avg),
                                       (v_corr_d, mag_corr_avg)):
                        _acc[:] = 0.0
                        for _i in range(3):
                            np.square(_src[_i, ..., _t], out=_tmp,
                                      dtype=np.float32)
                            _acc += _tmp
                        np.sqrt(_acc, out=_acc)
                        _dst += _acc
                    # |v_corr - v_meas|, without ever forming `delta`
                    _acc[:] = 0.0
                    for _i in range(3):
                        np.subtract(v_corr_d[_i, ..., _t],
                                    v_dcm[_i, ..., _t], out=_tmp,
                                    dtype=np.float32)
                        np.multiply(_tmp, _tmp, out=_tmp)
                        _acc += _tmp
                    np.sqrt(_acc, out=_acc)
                    mag_delta_avg += _acc
                mag_meas_avg  /= _Ntd
                mag_corr_avg  /= _Ntd
                mag_delta_avg /= _Ntd
                del _acc, _tmp

                # Apply spatial mask
                mask = inside_r0_d if inside_r0_d is not None else \
                       np.ones(mag_delta_avg.shape, dtype=bool)

                m_in   = mag_meas_avg[mask]   # measured velocity, inside R0
                c_in   = mag_corr_avg[mask]   # corrected velocity, inside R0
                d_in   = mag_delta_avg[mask]  # correction magnitude, inside R0

                # Key statistics
                stats = {
                    # Correction magnitude |v_cor − v_meas|
                    "delta_mean_cm_s":   float(d_in.mean()           * 100),
                    "delta_median_cm_s": float(np.median(d_in)       * 100),
                    "delta_p95_cm_s":    float(np.percentile(d_in,95)* 100),
                    "delta_max_cm_s":    float(d_in.max()            * 100),
                    # Velocity magnitudes before/after (confirms data loaded)
                    "vmeas_mean_cm_s":   float(m_in.mean()           * 100),
                    "vcorr_mean_cm_s":   float(c_in.mean()           * 100),
                    "vmeas_p99_cm_s":    float(np.percentile(m_in,99)* 100),
                    "vcorr_p99_cm_s":    float(np.percentile(c_in,99)* 100),
                    # Relative change: what fraction of the signal was corrected
                    "pct_signal_corrected": float(
                        100.0 * d_in.mean() / max(m_in.mean(), 1e-9)),
                    "n_voxels_corrected": int(mask.sum()),
                }

                print("\n  [DICOM] ── Correction results (inside r ≤ R₀) ──────────")
                print("  [DICOM] Correction magnitude |v_cor − v_meas|:")
                print(f"           mean   = {stats['delta_mean_cm_s']:.3f} cm/s")
                print(f"           median = {stats['delta_median_cm_s']:.3f} cm/s")
                print(f"           p95    = {stats['delta_p95_cm_s']:.3f} cm/s")
                print(f"           max    = {stats['delta_max_cm_s']:.3f} cm/s")
                _n_alias = int((mag_meas_avg[mask] > VENC_d).sum()) if mask.any() else 0
                if _n_alias > 0:
                    print(f"  [DICOM] NOTE: {_n_alias:,} masked voxels "
                          f"({100*_n_alias/max(int(mask.sum()),1):.2f}%) exceed VENC "
                          f"({VENC_d*100:.0f} cm/s). |v| above VENC cannot be measured "
                          "directly — these are aliased or residual dropout noise.")
                print("  [DICOM] Velocity magnitude (mean over signal voxels in R₀, time-avg):")
                print(f"           before correction: {stats['vmeas_mean_cm_s']:.2f} cm/s")
                print(f"           after  correction: {stats['vcorr_mean_cm_s']:.2f} cm/s")
                print("  [DICOM] Mean relative correction "
                      f"(mean|Δv| / mean|v|): {stats['pct_signal_corrected']:.1f}%")
                if signal_only_frac is not None:
                    print(f"  [DICOM] Voxels corrected: {stats['n_voxels_corrected']:,}"
                          f"  (inside R₀: {100*r0_only_frac:.0f}%"
                          f"  ·  with signal: {100*signal_only_frac:.0f}%)")
                else:
                    print(f"  [DICOM] Voxels corrected: {stats['n_voxels_corrected']:,}"
                          f"  (inside R₀: {100*r0_only_frac:.0f}%)")
                print("  [DICOM] NOTE: No ground truth available for acquired data.")
                print(f"           Use {_fig_label(19).strip('[]')} to visually "
                      "inspect before/after correction.")
                print("           Validate by checking spatial correlation between")
                print(f"           |v_cor−v_meas| map ({_fig_label(19).strip('[]')}) and "
                      f"Λ deviation map ({_fig_label(2).strip('[]')}).")

                # ── Vessel-restricted statistics ──────────────────────────
                # The magnitude mask admits all tissue with signal, but in
                # parenchyma the three velocity components are pure noise and
                # |v| = sqrt(vx²+vy²+vz²) rectifies that noise into a nonzero
                # speed. Averaging the correction over those voxels dilutes it,
                # because |Δv| ≈ ‖Λ⁻ᵀ−I‖·|v| scales with the LOCAL SPEED: a
                # 6 cm/s noise voxel yields a correction ~7× smaller than a
                # 45 cm/s arterial voxel. The physically meaningful number is
                # the correction on voxels that actually carry blood flow.
                #
                # The threshold is set from the noise floor itself: in
                # stationary tissue |v| follows a Maxwell distribution with
                # median = 1.5382σ, so σ is estimated robustly from the median
                # and the vessel cut is placed at N·σ (default 4σ).
                # ── Temporal-coherence gate ───────────────────────────────
                # Computed AFTER the background fit on purpose: that fit needs
                # the static compartment, which is exactly what this gate
                # removes. Order matters here.
                _flow_gate = None
                if (DICOM_FLOW_COHERENCE_MIN is not None
                        and v_dcm.ndim == 5 and v_dcm.shape[-1] >= 4):
                    try:
                        _vt_mean = v_dcm.mean(axis=-1)                 # (3,Nx,Ny,Nz)
                        _coh_num = np.sqrt((_vt_mean ** 2).sum(axis=0))
                        _coh = _coh_num / np.maximum(mag_meas_avg, 1e-12)

                        # Coherence measures temporal CONSISTENCY, not motion.
                        # A voxel that is barely moving but moves the same
                        # negligible amount every phase has |mean_t(v)| equal
                        # to mean_t(|v|) and so scores coherence ~1 — it
                        # passes a coherence-only gate. That is exactly what
                        # the interior of the static sphere looks like after
                        # the background plane has been subtracted from it:
                        # a small, smooth, time-invariant residual.
                        #
                        # Measured consequence on Series 11-9: the gated mask
                        # filled with near-zero constant voxels, its median
                        # speed collapsed to 0.03 cm/s, and the noise estimate
                        # sigma = median/1.5382 fell from 3.12 to 0.018 cm/s —
                        # an implied SNR of 3666, which is not physical.
                        #
                        # A speed floor fixes it. This floor is deliberately
                        # LOW (1 sigma): its only job is to reject the
                        # not-moving-at-all population. The vessel threshold
                        # downstream does the real selection.
                        _sig_pre = float(np.median(mag_meas_avg[mask]) / 1.5382) \
                            if mask is not None else \
                            float(np.median(mag_meas_avg) / 1.5382)
                        _fs = float(DICOM_COHERENCE_SPEED_FLOOR_SIGMA)
                        if _fs <= 0.0:
                            # No speed selection anywhere: the coherence cut
                            # alone separates flowing from static fluid. Slow
                            # flow is kept by construction, and so is anything
                            # stationary that happens to be coherent - the
                            # noise-consistent fraction reported below is the
                            # honest measure of how much that is.
                            _spd_floor = 0.0
                        else:
                            _spd_floor = max(_fs * _sig_pre,
                                             0.005 * float(VENC_d))
                        _flow_gate = ((_coh >= float(DICOM_FLOW_COHERENCE_MIN))
                                      & (mag_meas_avg > _spd_floor))
                        if mask is not None:
                            _n_before = int((mask).sum())
                            _n_after = int((mask & _flow_gate).sum())
                        else:
                            _n_before = int(_flow_gate.size)
                            _n_after = int(_flow_gate.sum())
                        _exp_static = 1.0 / np.sqrt(max(v_dcm.shape[-1], 1))
                        print("\n  [DICOM] ── Flow-coherence gate "
                              "(separates flowing from static fluid) ──")
                        print(f"           coherence = |mean_t(v)| / mean_t(|v|), "
                              f"cut at {DICOM_FLOW_COHERENCE_MIN:g}")
                        print(f"           static fluid predicts ~1/sqrt(Nt) = "
                              f"{_exp_static:.2f}; real flow ~1")
                        print(f"           plus a speed floor of "
                              f"{_spd_floor*100:.2f} cm/s ({_fs:g} sigma) — coherence "
                              "alone also passes\n           voxels that are "
                              "consistently NOT moving"
                              if _spd_floor > 0.0 else
                              "           NO speed floor "
                              "(DICOM_COHERENCE_SPEED_FLOOR_SIGMA = 0):\n"
                              "           selection is coherence only, so slow "
                              "flow is kept and no velocity\n"
                              "           threshold is applied anywhere in the "
                              "pipeline")
                        print(f"           kept {_n_after:,} of {_n_before:,} "
                              f"masked voxels ({100.0*_n_after/max(_n_before,1):.1f}%)")
                        if _n_after < 200:
                            print("           [WARN] almost nothing survived — the "
                                  "cut is too strict for this flow, or the flow "
                                  "reverses during the cycle. Gate DISABLED.")
                            _flow_gate = None
                        else:
                            # Keep the pre-gate mask: the noise floor must be
                            # estimated on the population that CONTAINS noise.
                            # Measuring sigma after the gate measures the
                            # spread of the surviving flow, which is not a
                            # noise estimate at all.
                            _mask_ungated = mask.copy() if mask is not None else None

                            # ── WHICH GATE REMOVES WHICH VOXELS ───────────
                            # Attribution, not a guess. A lumen that appears
                            # only partly filled in the correction panels can
                            # be cut by the coherence test, by the speed
                            # floor, or by neither (an oblique cross-section
                            # through a tilted tube). Those look identical in
                            # the figure and have completely different
                            # meanings, so the counts are reported per gate
                            # and the CENTROID SHIFT is measured: a symmetric
                            # threshold on a symmetric profile removes a ring
                            # and leaves the centroid where it was, whereas a
                            # chord removed from one side moves it.
                            try:
                                from scipy import ndimage as _nd
                                _lab_m, _nl_m = _nd.label(_mask_ungated)
                                if _nl_m:
                                    _cnt = np.bincount(_lab_m.ravel())
                                    _cnt[0] = 0
                                    _big = int(_cnt.argmax())
                                    _lu = (_lab_m == _big)
                                    _n_lu = int(_lu.sum())
                                    _pc = int((_lu & (_coh >= float(
                                        DICOM_FLOW_COHERENCE_MIN))).sum())
                                    _pf = int((_lu & (mag_meas_avg
                                                      > _spd_floor)).sum())
                                    _pb = int((_lu & _flow_gate).sum())
                                    _c0 = np.argwhere(_lu).mean(axis=0)
                                    _g = _lu & _flow_gate
                                    _c1 = (np.argwhere(_g).mean(axis=0)
                                           if _g.any() else _c0)
                                    _shift = np.abs(_c1 - _c0)
                                    print("\n  [DICOM] ── Gate attribution, "
                                          "largest connected structure ──")
                                    print(f"           structure       "
                                          f"{_n_lu:9,} voxels")
                                    print(f"           passes coherence"
                                          f"{_pc:9,}  ({100*_pc/max(_n_lu,1):5.1f}%)")
                                    print(f"           passes speed floor"
                                          f"{_pf:7,}  ({100*_pf/max(_n_lu,1):5.1f}%)")
                                    print(f"           passes both     "
                                          f"{_pb:9,}  ({100*_pb/max(_n_lu,1):5.1f}%)")
                                    print(f"           centroid shift  "
                                          f"({_shift[0]:.1f}, {_shift[1]:.1f}, "
                                          f"{_shift[2]:.1f}) voxels")
                                    if float(_shift.max()) > 1.5:
                                        print("           ASYMMETRIC: the gate "
                                              "removed more from one side than "
                                              "the other, so this is")
                                        print("           not a wall annulus. "
                                              "Suspect an oblique cross-section, "
                                              "a magnitude shading")
                                        print("           gradient, or genuinely "
                                              "one-sided flow.")
                                    else:
                                        print("           symmetric: consistent "
                                              "with a wall annulus falling below "
                                              "the threshold.")
                            except Exception as _ge:
                                print(f"  [DICOM] gate attribution skipped "
                                      f"({type(_ge).__name__}: {_ge})")

                            # ── CORE vs ANNULUS, PHASE BY PHASE ──────────
                            # A lumen that shows a fast ring with a slower
                            # centre has three quite different explanations
                            # and they look identical in one still frame:
                            #
                            #  (a) genuine annular (Womersley) flow — in
                            #      strongly pulsatile flow the peak migrates
                            #      toward the wall during part of the cycle,
                            #      so the ratio is < 1 at SOME phases only;
                            #  (b) velocity aliasing — the core exceeds VENC,
                            #      wraps, and reads LOW at every phase where
                            #      it is fastest;
                            #  (c) intravoxel dephasing — turbulence or shear
                            #      destroys signal in the core, which shows up
                            #      as a MAGNITUDE void, not just a slow speed.
                            #
                            # Measuring the ratio per phase separates (a) from
                            # (b), and reporting the core magnitude separates
                            # (c) from both. None of this changes the
                            # correction — Lambda is applied voxel-wise
                            # regardless — but it decides which voxels pass
                            # the vessel threshold and therefore enter the
                            # reported statistics.
                            try:
                                if (_nl_m and v_dcm.ndim == 5
                                        and mag_for_plot is not None):
                                    _idx = np.argwhere(_lu)
                                    _c = _idx.mean(axis=0)
                                    _rad = np.linalg.norm(
                                        _idx - _c, axis=1)
                                    _rmax = float(np.percentile(_rad, 98))
                                    _core = np.zeros_like(_lu)
                                    _ann = np.zeros_like(_lu)
                                    _core[tuple(_idx[_rad < 0.40*_rmax].T)] = True
                                    _ann[tuple(_idx[(_rad > 0.55*_rmax)
                                                    & (_rad < 0.90*_rmax)].T)] = True
                                    if _core.sum() > 20 and _ann.sum() > 20:
                                        print("\n  [DICOM] ── Lumen profile, "
                                              "core vs annulus, per phase ──")
                                        print("           phase   core   "
                                              "annulus   ratio   [cm/s]")
                                        _ratios = []
                                        for _t in range(v_dcm.shape[-1]):
                                            _sp = np.sqrt((v_dcm[..., _t] ** 2
                                                           ).sum(axis=0))
                                            _kc = float(_sp[_core].mean()*100)
                                            _ka = float(_sp[_ann].mean()*100)
                                            _rt = _kc / max(_ka, 1e-9)
                                            _ratios.append(_rt)
                                            print(f"           {_t:5d}  "
                                                  f"{_kc:6.1f}  {_ka:7.1f}   "
                                                  f"{_rt:5.2f}")
                                        _mn, _mx = min(_ratios), max(_ratios)
                                        _magc = float(np.mean(
                                            mag_for_plot[_core]))
                                        _maga = float(np.mean(
                                            mag_for_plot[_ann]))
                                        print(f"           ratio spans "
                                              f"{_mn:.2f}–{_mx:.2f} over the "
                                              f"cycle;  core/annulus MAGNITUDE"
                                              f" = {_magc/max(_maga,1e-9):.2f}")
                                        if _magc / max(_maga, 1e-9) < 0.7:
                                            print("           the core is DARK "
                                                  "in the magnitude image → "
                                                  "signal loss (dephasing),")
                                            print("           not a velocity "
                                                  "measurement error.")
                                        elif (_mx - _mn) > 0.25 and _mn < 1.05:
                                            print("           the profile "
                                                  "INVERTS during the cycle → "
                                                  "annular (Womersley) flow,")
                                            print("           which is real "
                                                  "phantom physics, not an "
                                                  "artefact.")
                                        elif _mx < 0.95:
                                            print("           the core is slow "
                                                  "at EVERY phase → suspect "
                                                  "aliasing: check whether the")
                                            print("           core exceeds "
                                                  "VENC, and re-acquire with a "
                                                  "higher VENC if so.")
                            except Exception as _pe:
                                print(f"  [DICOM] lumen profile skipped "
                                      f"({type(_pe).__name__}: {_pe})")

                            mask = mask & _flow_gate
                    except Exception as _exc:
                        print("  [DICOM] Flow-coherence gate skipped "
                              f"({type(_exc).__name__}: {_exc})")
                        _flow_gate = None

                vessel_stats = None
                _vthr_for_plot = None
                _sigma_for_plot = None
                try:
                    _sig_mask = (_mask_ungated
                                 if ("_mask_ungated" in dir() and _mask_ungated is not None)
                                 else mask)
                    _sp = mag_meas_avg[_sig_mask]
                    _sigma = float(np.median(_sp) / 1.5382)
                    if (DICOM_VESSEL_THRESHOLD_CMS is not None
                            and float(DICOM_VESSEL_THRESHOLD_CMS) <= 0.0):
                        # NO velocity gate — analysis set = magnitude/brain
                        # mask. Slow vessels are kept by construction; so is
                        # parenchyma, whose apparent speed is rectified noise.
                        # The noise-consistent fraction is reported below.
                        _vthr = 0.0
                        _thr_src = ("DISABLED — no velocity gate; analysis "
                                    "set is the magnitude/brain mask")
                        _thr_field = 0.0
                    elif DICOM_VESSEL_THRESHOLD_CMS is not None:
                        _vthr = float(DICOM_VESSEL_THRESHOLD_CMS) / 100.0
                        _thr_src = "manual"
                        _thr_field = _vthr
                    elif sigma_v_map is not None:
                        # Spatially varying cut: k·σ_v(r). Low-SNR scalp, orbit
                        # and skull-base voxels get a proportionally HIGHER bar,
                        # so their noise is rejected without discarding genuine
                        # vessels in high-SNR brain.
                        _thr_field = np.maximum(
                            DICOM_VESSEL_SIGMA_MULT * sigma_v_map,
                            DICOM_VESSEL_MIN_VENC_FRAC * float(hdr.get("VENC", cfg.VENC)))
                        _vthr = float(np.median(_thr_field[mask]))
                        _thr_src = (f"auto = {DICOM_VESSEL_SIGMA_MULT:g}·σ_v(r), "
                                    "per-voxel from local SNR "
                                    f"(median {_vthr*100:.1f} cm/s)")
                    else:
                        _vthr = max(DICOM_VESSEL_SIGMA_MULT * _sigma,
                                    DICOM_VESSEL_MIN_VENC_FRAC * VENC_d)
                        _thr_src = (f"auto = max({DICOM_VESSEL_SIGMA_MULT:g}σ, "
                                    "15 cm/s)  [global — no SNR map]")
                        _thr_field = _vthr
                    # DISPLAY floor is deliberately NOT the statistics
                    # threshold. With the gate disabled a zero display floor
                    # would paint the whole brain in the fig05 MIP, hiding the
                    # vessels the figure exists to show, so the figure keeps a
                    # noise-based floor while the STATISTICS use every voxel.
                    if np.ndim(_thr_field) == 0 and float(_thr_field) <= 0.0:
                        _vthr_for_plot = (
                            np.maximum(DICOM_VESSEL_SIGMA_MULT * sigma_v_map,
                                       DICOM_VESSEL_MIN_VENC_FRAC
                                       * float(hdr.get("VENC", cfg.VENC)))
                            if sigma_v_map is not None
                            else max(DICOM_VESSEL_SIGMA_MULT * _sigma,
                                     DICOM_VESSEL_MIN_VENC_FRAC * VENC_d))
                    else:
                        _vthr_for_plot = _thr_field
                    _sigma_for_plot = sigma_v_map if sigma_v_map is not None else _sigma
                    vessel = (mask.copy() if (np.ndim(_thr_field) == 0
                                              and float(_thr_field) <= 0.0)
                              else mask & (mag_meas_avg > _thr_field))
                    n_ves = int(vessel.sum())
                    print("\n  [DICOM] ── Vessel-restricted results ─────────────────")
                    print("  [DICOM] Velocity noise floor (Maxwell fit to tissue):")
                    print(f"           σ = {_sigma*100:.2f} cm/s "
                          f"({100*_sigma/max(VENC_d,1e-9):.1f}% of VENC)  ·  "
                          "implied SNR ≈ "
                          f"{np.sqrt(2)*VENC_d/(np.pi*max(_sigma,1e-9)):.0f}"
                          " (linear) = "
                          f"{20*np.log10(np.sqrt(2)*VENC_d/(np.pi*max(_sigma,1e-9))):.0f}"
                          " dB")
                    # UNITS. The ratio above is LINEAR; cfg.snr_db is in
                    # decibels. Printing the bare linear number next to a
                    # config field named snr_db invited reading 13 as 13 dB
                    # when it means 22 dB, so both are now shown.
                    if _vthr > 0:
                        print(f"  [DICOM] Vessel threshold = {_vthr*100:.1f} "
                              f"cm/s ({_thr_src})")
                    else:
                        print(f"  [DICOM] Vessel threshold: {_thr_src}")

                    # ── How much of the analysis set is rectified NOISE? ──
                    # |v| = sqrt(vx^2+vy^2+vz^2) of three Gaussian noise
                    # components follows a Maxwell distribution with
                    #     mean   = sqrt(8/pi)*sigma ~ 1.596*sigma
                    #     median = 1.5382*sigma
                    # so stationary tissue shows a POSITIVE apparent speed
                    # that averaging cannot remove. Comparing the observed
                    # speeds against that prediction says what fraction of the
                    # reported set could be tissue rather than blood. This is
                    # reporting only -- nothing is excluded.
                    try:
                        _sig_n = (float(np.median(sigma_v_map[vessel]))
                                  if sigma_v_map is not None
                                  else float(_sigma))
                        _sp_v = mag_meas_avg[vessel]
                        _noise_mean = np.sqrt(8.0 / np.pi) * _sig_n
                        _frac_below = float((_sp_v <= _noise_mean).mean())
                        vessel_stats_noise = {
                            "sigma_v_cm_s":        _sig_n * 100,
                            "noise_mean_speed_cm_s": _noise_mean * 100,
                            "frac_at_or_below_noise": _frac_below,
                        }
                        print("  [DICOM] Noise-consistency of the analysis "
                              "set:")
                        print(f"           sigma_v = {_sig_n*100:.2f} cm/s  ->  "
                              "pure-noise |v| would average "
                              f"{_noise_mean*100:.2f} cm/s")
                        print(f"           {100*_frac_below:.1f}% of the "
                              "analysis voxels are at or below that level")
                        if _frac_below > 0.5:
                            print("           i.e. the MAJORITY of this set is "
                                  "consistent with stationary tissue.")
                            print("           The mean relative correction "
                                  "below is therefore dominated by the")
                            print("           correction applied to NOISE, not "
                                  "to blood. Quote it as a whole-mask")
                            print("           number, and report a "
                                  "flow-selected subset (temporal coherence,")
                            print("           DICOM_FLOW_COHERENCE_MIN) "
                                  "alongside it.")
                    except Exception as _ne:
                        vessel_stats_noise = None
                        print(f"  [DICOM] noise-consistency check skipped: {_ne}")

                    # ── Does the SIMULATION match this data? ─────────────
                    #
                    # cfg.snr_db governs the synthetic experiment reported in
                    # fig02/fig04, and the percentage improvement there is
                    # dominated by it: the residual after correction is
                    # essentially pure noise, so the same correction reads
                    # very differently at 50 dB and at the ~20 dB typical of
                    # acquired data (see the SNR sweep). The DICOM stage has
                    # just measured the real noise floor, so the mismatch can
                    # be detected rather than left to the reader to notice.
                    # This is reporting only — nothing is altered.
                    try:
                        # WHICH sigma. Two independent estimates exist:
                        #   (a) _sigma      — Maxwell fit to the velocity
                        #       spread of the "static tissue" population.
                        #       Assumes such a population EXISTS. In a pipe
                        #       rig where 85% of signal voxels are flowing it
                        #       is contaminated by flow and overestimates
                        #       badly: 16.56 cm/s (implying 4 dB) against
                        #   (b) sigma_v_map — from the per-voxel magnitude
                        #       SNR via sigma_v = sqrt(2)*VENC/(pi*SNR), which
                        #       needs no static population at all: 4.2 cm/s,
                        #       implying 16 dB.
                        # Prefer (b) when it exists, and say so when they
                        # disagree — a factor-4 gap is itself a finding about
                        # the static mask, not a noise measurement.
                        _sig_fit = float(_sigma)
                        _sig_map = (float(np.median(sigma_v_map[mask]))
                                    if sigma_v_map is not None
                                    and mask is not None
                                    and np.any(mask) else None)
                        _sig_use = _sig_map if _sig_map else _sig_fit
                        _src_use = ("local-SNR map" if _sig_map
                                    else "Maxwell fit to static tissue")
                        if _sig_map and _sig_fit / max(_sig_map, 1e-9) > 2.0:
                            print(f"  [DICOM] NOTE: the two noise estimates "
                                  f"disagree by {_sig_fit/_sig_map:.1f}x "
                                  f"(static-fit {_sig_fit*100:.2f} vs "
                                  f"local-SNR {_sig_map*100:.2f} cm/s).")
                            print("           The static-tissue fit needs a "
                                  "genuinely static population; with most "
                                  "signal")
                            print("           voxels flowing it measures flow, "
                                  "not noise. Using the local-SNR value below.")
                        _snr_data = 20.0 * np.log10(
                            np.sqrt(2) * VENC_d / (np.pi * max(_sig_use, 1e-9)))
                        _d_snr = float(cfg.snr_db) - _snr_data
                        # Record the measurement regardless of verbosity — the
                        # console banner can be quietened, but the number is a
                        # property of the DATA and stays in run_config.json so
                        # the run remains reproducible and self-documenting.
                        #
                        # `vessel_stats` is still None at this point: it is not
                        # constructed until the vessel-restricted block far
                        # below. The previous version assigned into it here
                        # inside `except Exception: pass`, so every one of
                        # these keys was silently discarded on every run and
                        # measured_snr_db never appeared in run_config.json.
                        # Buffer them instead and merge once the dict exists.
                        _snr_record = {
                            "measured_snr_db":      float(_snr_data),
                            "measured_sigma_v_cm_s": float(_sig_use * 100),
                            "measured_sigma_source": _src_use,
                            "sim_snr_db":           float(cfg.snr_db),
                            "snr_matches_sim":      bool(abs(_d_snr) <= 3.0),
                        }
                        if abs(_d_snr) > 3.0 and SNR_MISMATCH_VERBOSE:
                            print()
                            print("  " + "!" * 68)
                            print(f"  [SIM/DATA MISMATCH] This data implies "
                                  f"SNR ≈ {_snr_data:.0f} dB, but the "
                                  f"simulation ran at {cfg.snr_db:.0f} dB.")
                            print(f"           sigma_v measured here = "
                                  f"{_sig_use*100:.2f} cm/s ({_src_use});  at "
                                  f"{cfg.snr_db:.0f} dB the simulation "
                                  "assumed")
                            print(f"           {np.sqrt(2)*VENC_d/(np.pi*10**(cfg.snr_db/20))*100:.2f}"
                                  " cm/s.")
                            print("           The % improvement in fig02 / "
                                  "fig04 is a NOISE statistic and does not")
                            print("           describe this dataset. Set "
                                  f"SimulationConfig.snr_db = {_snr_data:.0f} "
                                  f"(and VENC = {VENC_d:.2f}, Nt = "
                                  f"{v_dcm.shape[-1]}) and re-run,")
                            print("           or quote the SNR-independent "
                                  "noise-free row instead.")
                            print("  " + "!" * 68)
                        elif abs(_d_snr) > 3.0:
                            # Quiet mode: one factual line, no banner. The
                            # simulation SNR is a deliberate choice (high SNR
                            # keeps the SYSTEMATIC GNL error visible in
                            # fig02/fig04, which is what those figures exist to
                            # show); the noise dependence is reported by the
                            # SNR sweep and the noise-free row instead.
                            print(f"  [DICOM] Velocity noise floor implies "
                                  f"SNR ≈ {_snr_data:.0f} dB for this dataset "
                                  f"(simulation figures: {cfg.snr_db:.0f} dB "
                                  "— see the SNR sweep).")
                        else:
                            print(f"  [DICOM] Simulation SNR "
                                  f"({cfg.snr_db:.0f} dB) matches this data "
                                  f"(≈ {_snr_data:.0f} dB). ✓")
                    except Exception:
                        pass
                    if n_ves < 100:
                        print(f"  [DICOM] Only {n_ves} voxels above threshold — "
                              "too few for vessel statistics; skipping.")
                    else:
                        d_ves = mag_delta_avg[vessel]
                        m_ves = mag_meas_avg[vessel]
                        vessel_stats = {
                            "vessel_threshold_cm_s": float(_vthr * 100),
                            "velocity_gate_disabled": bool(_vthr <= 0),
                            "noise_consistency": (
                                vessel_stats_noise
                                if "vessel_stats_noise" in dir() else None),
                            "vessel_sigma_cm_s":     float(_sigma * 100),
                            "n_vessel_voxels":       n_ves,
                            # NAMING (fixed). `mask` has by this point been
                            # narrowed by the flow-coherence gate on the
                            # phantom/pipe presets (mask &= _flow_gate above),
                            # so this ratio is vessel / FLOW-GATED voxels, not
                            # vessel / signal voxels. On the ideal phantom it
                            # read 0.567, i.e. "57% of signal voxels are
                            # vessel", when the true share of the signal mask
                            # is 1.3%. The in-vivo preset has the gate off, so
                            # there the old name happened to be correct — which
                            # is exactly how it survived. Both are now given.
                            "vessel_frac_of_analysis_mask": float(
                                n_ves / max(int(mask.sum()), 1)),
                            "vessel_frac_of_signal": float(
                                n_ves / max(int(_sig_mask.sum()), 1)
                                if _sig_mask is not None else float("nan")),
                            "analysis_mask_n": int(mask.sum()),
                            "signal_mask_n": (int(_sig_mask.sum())
                                              if _sig_mask is not None else None),
                            "delta_mean_cm_s":   float(d_ves.mean()            * 100),
                            "delta_median_cm_s": float(np.median(d_ves)        * 100),
                            "delta_p95_cm_s":    float(np.percentile(d_ves,95) * 100),
                            "vmeas_mean_cm_s":   float(m_ves.mean()            * 100),
                            "pct_signal_corrected": float(
                                100.0 * d_ves.mean() / max(m_ves.mean(), 1e-9)),
                            # Same quantity as correction_magnitude_cm_s.
                            # pct_signal_corrected but over a DIFFERENT voxel
                            # set (vessels only vs all signal voxels), so the
                            # two legitimately differ. Aliased under an
                            # unambiguous name; the original key is kept for
                            # backward compatibility with earlier run logs.
                            "pct_vessel_corrected": float(
                                100.0 * d_ves.mean() / max(m_ves.mean(), 1e-9)),
                        }
                        # Merge the noise/SNR measurement buffered earlier (see
                        # the _snr_record comment above). Guarded with dir() so
                        # a run that never reached the noise-estimation branch
                        # still produces a valid vessel_stats block.
                        if "_snr_record" in dir():
                            vessel_stats.update(_snr_record)
                        print(f"  [DICOM] Analysis voxels: {n_ves:,} "
                              f"({100*vessel_stats['vessel_frac_of_signal']:.2f}% "
                              "of signal voxels)"
                              + ("  [no velocity gate]" if _vthr <= 0 else ""))
                        print("  [DICOM] Mean |v| on vessels: "
                              f"{vessel_stats['vmeas_mean_cm_s']:.1f} cm/s "
                              f"(vs {stats['vmeas_mean_cm_s']:.1f} cm/s over all "
                              "signal voxels)")
                        # ── Flow-selected SUBSET (reporting only) ────────
                        # Temporal coherence separates flow from noise where a
                        # speed threshold cannot: real flow holds its direction
                        # across the cycle, noise averages down as 1/sqrt(Nt).
                        # A slow but genuine vessel is KEPT; a faster-looking
                        # noise voxel is not. The primary analysis set above is
                        # untouched — this only adds a second number.
                        if (DICOM_COHERENT_SUBSET_MIN is not None
                                and v_dcm.ndim == 5 and v_dcm.shape[-1] >= 4):
                            try:
                                _vtm = v_dcm.mean(axis=-1)
                                _cohs = (np.sqrt((_vtm ** 2).sum(axis=0))
                                         / np.maximum(mag_meas_avg, 1e-12))
                                _sub = vessel & (
                                    _cohs >= float(DICOM_COHERENT_SUBSET_MIN))
                                _nsub = int(_sub.sum())
                                _exp_noise = 1.0/np.sqrt(v_dcm.shape[-1])
                                print("\n  [DICOM] ── Flow-selected subset "
                                      "(temporal coherence) ──")
                                print("           coherence = |mean_t(v)| / "
                                      "mean_t(|v|), cut at "
                                      f"{DICOM_COHERENT_SUBSET_MIN:g}  "
                                      f"(noise predicts ~{_exp_noise:.2f})")
                                if _nsub >= 100:
                                    _ds = mag_delta_avg[_sub]
                                    _ms = mag_meas_avg[_sub]
                                    _rel = float(100.0*_ds.mean()
                                                 / max(_ms.mean(), 1e-9))
                                    vessel_stats["coherent_subset"] = {
                                        "coherence_min": float(
                                            DICOM_COHERENT_SUBSET_MIN),
                                        "n_voxels": _nsub,
                                        "frac_of_analysis_set": float(
                                            _nsub / max(n_ves, 1)),
                                        "vmeas_mean_cm_s": float(_ms.mean()*100),
                                        "vmeas_min_cm_s":  float(_ms.min()*100),
                                        "delta_mean_cm_s": float(_ds.mean()*100),
                                        "delta_median_cm_s": float(
                                            np.median(_ds)*100),
                                        "delta_p95_cm_s": float(
                                            np.percentile(_ds, 95)*100),
                                        "pct_corrected": _rel,
                                    }
                                    print(f"           n = {_nsub:,} voxels "
                                          f"({100*_nsub/max(n_ves,1):.2f}% of "
                                          "the analysis set)")
                                    print("           mean |v| = "
                                          f"{_ms.mean()*100:.1f} cm/s   "
                                          "slowest kept = "
                                          f"{_ms.min()*100:.1f} cm/s "
                                          "(slow vessels retained)")
                                    print("           correction: mean "
                                          f"{_ds.mean()*100:.3f}  median "
                                          f"{np.median(_ds)*100:.3f}  p95 "
                                          f"{np.percentile(_ds,95)*100:.3f} "
                                          "cm/s")
                                    print("           mean relative correction "
                                          f"= {_rel:.1f}%")
                                    print("           ^ quote THIS as the "
                                          "correction on flowing blood; the "
                                          "whole-mask")
                                    print("             block above is the "
                                          "inclusive number.")
                                else:
                                    print(f"           only {_nsub} voxels "
                                          "survived — too few to report; "
                                          "lower the cut.")
                            except Exception as _ce:
                                print("  [DICOM] coherent-subset block skipped "
                                      f"({type(_ce).__name__}: {_ce})")

                        print("  [DICOM] Correction on vessels |v_cor − v_meas|:")
                        print(f"           mean   = {vessel_stats['delta_mean_cm_s']:.3f} cm/s")
                        print(f"           median = {vessel_stats['delta_median_cm_s']:.3f} cm/s")
                        print(f"           p95    = {vessel_stats['delta_p95_cm_s']:.3f} cm/s")
                        print("  [DICOM] Mean relative correction on vessels "
                              "(mean|Δv| / mean|v|): "
                              f"{vessel_stats['pct_signal_corrected']:.1f}%")

                        # ── relative correction vs radius ────────────────
                        # |dv| ~ |(Lambda-I) v| carries BOTH the geometry term
                        # and the local speed, and in a phantom the speed
                        # varies far more than the position does — so |dv| by
                        # itself largely tracks where the fluid is fast, and
                        # its correlation with the Lambda map understates the
                        # geometric effect. Dividing by |v| removes the speed
                        # factor and leaves the part that depends only on
                        # position, which is what the gradient model predicts.
                        try:
                            _rr = np.sqrt(Xd**2 + Yd**2 + Zd**2)[vessel] * 1000.0
                            _rel = 100.0 * d_ves / np.maximum(m_ves, 1e-9)
                            _ok = np.isfinite(_rr) & np.isfinite(_rel)
                            if _ok.sum() > 100:
                                _c_rel = float(np.corrcoef(_rr[_ok], _rel[_ok])[0, 1])
                                _c_abs = float(np.corrcoef(_rr[_ok],
                                                           d_ves[_ok])[0, 1])
                                _c_spd = float(np.corrcoef(m_ves[_ok],
                                                           d_ves[_ok])[0, 1])
                                vessel_stats["corr_relcorr_vs_radius"] = _c_rel
                                vessel_stats["corr_abscorr_vs_radius"] = _c_abs
                                vessel_stats["corr_abscorr_vs_speed"]  = _c_spd
                                print("\n  [DICOM] ── Relative correction "
                                      "|Δv|/|v| (isolates the geometry) ──")
                                print(f"           corr(|Δv|/|v|, radius) = "
                                      f"{_c_rel:+.3f}   <- report this")
                                print(f"           corr(|Δv|,     radius) = "
                                      f"{_c_abs:+.3f}")
                                print(f"           corr(|Δv|,     speed ) = "
                                      f"{_c_spd:+.3f}   (why |Δv| alone "
                                      "understates it)")
                                _edges = [20, 40, 60, 80, 100, 130, 200]
                                _prof = []
                                print("             r [mm]     |Δv| cm/s   "
                                      "|Δv|/|v| %        n")
                                for _lo, _hi in zip(_edges[:-1], _edges[1:]):
                                    _m = (_rr >= _lo) & (_rr < _hi)
                                    if _m.sum() > 50:
                                        _a = float(d_ves[_m].mean() * 100)
                                        _b = float(_rel[_m].mean())
                                        _prof.append({"r_lo_mm": _lo,
                                                      "r_hi_mm": _hi,
                                                      "abs_cm_s": _a,
                                                      "rel_pct": _b,
                                                      "n": int(_m.sum())})
                                        print(f"             {_lo:3d}-{_hi:3d}"
                                              f"      {_a:8.4f}      {_b:6.2f}"
                                              f"    {int(_m.sum()):>8,}")
                                vessel_stats["radial_profile"] = _prof
                        except Exception as _exc:
                            print(f"  [DICOM] Relative-correction metric "
                                  f"skipped ({type(_exc).__name__}: {_exc})")

                        # ── encoding-direction error ────────────────────
                        # The thesis names errors in "both the magnitude AND
                        # direction of encoded velocities". |Δv| above covers
                        # magnitude; the angle between v_meas and v_corr
                        # covers direction, on the acquired slab rather than
                        # at coil level. Noise rotates the vector randomly
                        # and averages out, GNL rotation is systematic — so
                        # the rise with radius is the model signature.
                        try:
                            direction_error_stats(
                                v_dcm, v_corr_d, vessel, Xd, Yd, Zd,
                                speed_min_m_s=_vthr, stats=vessel_stats)
                        except Exception as _exc:
                            print("  [DICOM] Direction-error metric skipped "
                                  f"({type(_exc).__name__}: {_exc})")

                        # ── flow rate: the one EXTERNAL ground truth ─────
                        # Everything else here compares the data with itself.
                        # Q through the lumen can be compared with the pump or
                        # flowmeter setting, and for steady incompressible flow
                        # it must also be constant along the tube — so the
                        # spread across planes tests the measurement even with
                        # no reference value to hand.
                        try:
                            _vm3 = (v_dcm.mean(axis=-1) if v_dcm.ndim == 5
                                    else v_dcm)
                            _vc3 = (v_corr_d.mean(axis=-1) if v_corr_d.ndim == 5
                                    else v_corr_d)
                            # Re-order the spatial axes to PATIENT order (x, y, z)
                            # so that "array axis i" and "velocity component i"
                            # mean the same thing. See array_axes_to_patient().
                            # Identity for an axial stack, so axial results are
                            # unchanged; essential for sagittal/coronal ones.
                            _vessel_f, _mask_f = vessel, mask
                            _sig_f, _sp_f = _sig_mask, (dx, dy, dz)
                            _pmap = array_axes_to_patient(locals().get("_iop", None))
                            if _pmap is None:
                                print("  [DICOM] WARNING: the stack is not axis-aligned "
                                      "(or its orientation is missing), so the flow-rate")
                                print("          and flux tests cannot map array axes to "
                                      "patient axes. Do not trust their axis choice.")
                            elif _pmap != [0, 1, 2]:
                                _tr = [int(a) for a in np.argsort(_pmap)]
                                _vm3 = np.ascontiguousarray(
                                    np.transpose(_vm3, [0] + [1 + a for a in _tr]))
                                _vc3 = np.ascontiguousarray(
                                    np.transpose(_vc3, [0] + [1 + a for a in _tr]))
                                _vessel_f = np.transpose(vessel, _tr)
                                _mask_f = np.transpose(mask, _tr)
                                _sig_f = (None if _sig_mask is None
                                          else np.transpose(_sig_mask, _tr))
                                _sp_f = tuple((dx, dy, dz)[a] for a in _tr)
                                print("  [DICOM] Stack axes (columns, rows, slices) run "
                                      "along patient ("
                                      + ", ".join("xyz"[q] for q in _pmap)
                                      + "): volume re-ordered to (x, y, z)")
                                print("          for the flow-rate and flux tests, so "
                                      "the plane normal and the velocity component agree.")
                            _ax = int(np.argmax([
                                np.abs(_vm3[i][_vessel_f]).mean() for i in range(3)]))
                            _q_b, _pl = dicom_flow_rate(_vm3, _vessel_f,
                                                        _sp_f, _ax)
                            _q_a, _ = dicom_flow_rate(_vc3, _vessel_f,
                                                      _sp_f, _ax)
                            if _q_b.size >= 3:
                                _qb, _qa = float(np.median(_q_b)), float(np.median(_q_a))
                                _cv_b = 100.0 * _q_b.std() / max(abs(_q_b.mean()), 1e-9)
                                _cv_a = 100.0 * _q_a.std() / max(abs(_q_a.mean()), 1e-9)
                                vessel_stats["flow_rate"] = {
                                    "normal_axis":        "xyz"[_ax],
                                    "n_planes":           int(_q_b.size),
                                    "Q_before_ml_min":    _qb,
                                    "Q_after_ml_min":     _qa,
                                    "plane_spread_before_pct": float(_cv_b),
                                    "plane_spread_after_pct":  float(_cv_a),
                                }
                                print("\n  [DICOM] ── Volumetric flow rate "
                                      "(compare with the pump setting) ──")
                                print(f"           through-plane axis : "
                                      f"{'xyz'[_ax]}   ({_q_b.size} planes)")
                                print(f"           Q before correction: "
                                      f"{_qb:9.1f} ml/min")
                                print(f"           Q after  correction: "
                                      f"{_qa:9.1f} ml/min   "
                                      f"({100*(_qa-_qb)/max(abs(_qb),1e-9):+.2f}%)")
                                print(f"           plane-to-plane spread: "
                                      f"{_cv_b:.1f}% → {_cv_a:.1f}%  "
                                      "(steady flow should be constant along "
                                      "the tube)")
                                # Q is only meaningful through a SINGLE channel
                                # carrying flow one way. A phantom with several
                                # compartments, or a mask dominated by
                                # noise-level fluid, gives planes that sum
                                # unrelated flows — the total then varies
                                # wildly and can change sign, and the number
                                # means nothing. A large plane-to-plane spread
                                # is the symptom; say so instead of printing a
                                # figure that invites being quoted.
                                _qmin, _qmax = float(_q_b.min()), float(_q_b.max())
                                _signflip = _qmin * _qmax < 0
                                if _cv_b > 25.0 or _signflip:
                                    print()
                                    print("           NOT USABLE for this "
                                          "geometry:")
                                    print(f"             Q ranges {_qmin:.0f} "
                                          f"to {_qmax:.0f} ml/min across planes"
                                          + (" and CHANGES SIGN."
                                             if _signflip else "."))
                                    print("             Q is defined for one "
                                          "channel with flow in a single")
                                    print("             direction. A "
                                          "multi-compartment phantom, or a "
                                          "mask")
                                    print("             holding mostly "
                                          "noise-level fluid, breaks that. Do "
                                          "not")
                                    print("             quote this Q; the "
                                          "spread itself is the finding.")
                                    vessel_stats["flow_rate"]["usable"] = False
                                    vessel_stats["flow_rate"]["DO_NOT_QUOTE"] = (
                                        "Q is defined for a single channel "
                                        "with flow in one direction. This "
                                        "mask spans multiple vessels and Q "
                                        "changes sign across planes, so "
                                        "neither Q value is meaningful. The "
                                        "plane-to-plane spread is the "
                                        "finding, not the flow rate.")
                                else:
                                    vessel_stats["flow_rate"]["usable"] = True
                                    print("           A known pump/flowmeter "
                                          "value makes this an EXTERNAL check "
                                          "—")
                                    print("           the only one in the "
                                          "pipeline. Quote it if you have one.")
                            # ── PER-CHANNEL flux conservation ────────
                            # The block above integrates Q over the WHOLE
                            # vessel mask, which is only meaningful for a
                            # single lumen. Split the mask into channels and
                            # test each one separately: this is the pipeline's
                            # only Lambda-independent check (see
                            # flux_conservation).
                            _lumens = isolate_flow_lumens(
                                _sig_f if _sig_f is not None else _mask_f,
                                _vessel_f)
                            _flux = []
                            if _lumens:
                                print(f"\n  [DICOM] ── Flux conservation, "
                                      f"{len(_lumens)} channel(s) found ──")
                            for _li, _lu in enumerate(_lumens, 1):
                                _lm = _lu["mask"]
                                # The tube axis comes from the GEOMETRY (the
                                # direction along which the cross-section is
                                # steadiest), not from the dominant velocity
                                # component: with an oblique tube the latter
                                # picks a direction whose "cross-sections" are
                                # slanted cuts, and the resulting 62% area
                                # variation was read as a failed flux test.
                                _axg, _kp, _cvg = tube_axis_and_planes(_lm)
                                # Through-plane share of the speed for each of
                                # the three candidate plane normals, measured on
                                # the flow voxels of this channel.
                                _selm = (_lm & _vessel_f) if np.any(_lm & _vessel_f) else _lm
                                _spd_l = float(np.sqrt(
                                    (_vm3[:, _selm] ** 2).sum(axis=0)).mean())
                                _shares = [float(np.abs(_vm3[i][_selm]).mean()
                                                 / max(_spd_l, 1e-12))
                                           for i in range(3)]
                                _axv = int(np.argmax(_shares))
                                # The GEOMETRIC axis is preferred, because on an
                                # oblique tube the dominant-velocity axis cuts
                                # slanted cross-sections and the resulting area
                                # variation reads as a failed flux test.
                                #
                                # Geometry alone is NOT sufficient, though. It
                                # picks the direction along which the
                                # cross-section is steadiest, and for a channel
                                # that is short in one axis and long in another
                                # that is the SHORT axis - which may be
                                # orthogonal to the flow. On the ideal phantom
                                # this chose x for a channel 16 x 71 x 41 mm
                                # whose flow runs in y, leaving only 15% of the
                                # speed through-plane, and flux_conservation
                                # correctly refused the result: the only
                                # Lambda-independent validation in the pipeline
                                # was discarded because of the axis choice, not
                                # because of the data. Take the geometric axis
                                # only when the flow actually passes through its
                                # planes, using the same 0.30 threshold that
                                # flux_conservation enforces downstream.
                                if _axg is not None and _shares[_axg] >= 0.30:
                                    _axl = _axg
                                    if _axg != _axv:
                                        print(f"           channel {_li}: tube "
                                              f"axis {'xyz'[_axg]} (area CV "
                                              f"{_cvg:.0f}%, "
                                              f"{100*_shares[_axg]:.0f}% of "
                                              f"speed through-plane) chosen over "
                                              f"the dominant-velocity axis "
                                              f"{'xyz'[_axv]}")
                                else:
                                    _axl = _axv
                                    if _axg is not None:
                                        print(f"           channel {_li}: the "
                                              f"geometric axis {'xyz'[_axg]} "
                                              f"carries only "
                                              f"{100*_shares[_axg]:.0f}% of the "
                                              f"speed through-plane (below the "
                                              f"30% floor), so the flow-carrying "
                                              f"axis {'xyz'[_axv]} "
                                              f"({100*_shares[_axv]:.0f}%) is "
                                              f"used instead")
                                        if (str(DATASET_PRESET) == "pipe_flow"
                                                and _axg != _axv):
                                            _flags_on = bool(
                                                _axis_ok and (DICOM_DIRECTION_PERMUTE
                                                              or DICOM_DIRECTION_SIGNS))
                                            _print_axis_mismatch_banner(
                                                _axg, _axv, _shares[_axg], _shares[_axv],
                                                hdr.get("direction_axes"), _flags_on,
                                                (DICOM_DIRECTION_PERMUTE
                                                 if (_axis_ok and DICOM_DIRECTION_PERMUTE)
                                                 else None))
                                _fc = flux_conservation(_vm3, _vc3, _lm,
                                                        _sp_f, _axl)
                                _fc["lumen"] = _li
                                _fc["through_plane_shares"] = {
                                    "x": _shares[0], "y": _shares[1],
                                    "z": _shares[2]}
                                _fc["axis_source"] = (
                                    "geometry" if (_axg is not None
                                                   and _axl == _axg)
                                    else "dominant-velocity")
                                _fc["n_voxels"] = _lu["n_voxels"]
                                _fc["n_flow_voxels"] = _lu["n_flow_voxels"]
                                _fc["centroid_vox"] = list(_lu["centroid_vox"])
                                _flux.append(_fc)
                                if not _fc.get("usable"):
                                    print(f"           channel {_li}: not "
                                          f"usable — {_fc.get('why')}")
                                    continue
                                print(f"           channel {_li} "
                                      f"({_lu['n_voxels']:,} vox, axis "
                                      f"{_fc['normal_axis']}, "
                                      f"{_fc['n_planes']} planes):")
                                print(f"             Q  {_fc['Q_before_ml_min']:9.1f}"
                                      f" → {_fc['Q_after_ml_min']:9.1f} ml/min"
                                      f"   ({100*(_fc['Q_after_ml_min']-_fc['Q_before_ml_min'])/max(abs(_fc['Q_before_ml_min']),1e-9):+.2f}%)")
                                print(f"             CV {_fc['cv_before_pct']:8.2f}%"
                                      f" → {_fc['cv_after_pct']:8.2f}%"
                                      f"   ({_fc['cv_improvement_pct']:+.1f}% "
                                      "of the spread removed)")
                            if _flux:
                                vessel_stats["flux_conservation"] = _flux
                                if any(f.get("usable") for f in _flux):
                                    print("           A DROP in CV is "
                                          "Lambda-independent evidence: mass "
                                          "conservation is imposed by physics,")
                                    print("           not by the field model. "
                                          "A rise, or no change, is equally "
                                          "reportable — quote it either way.")
                        except Exception as _qe:
                            print(f"  [DICOM] Flow-rate check skipped "
                                  f"({type(_qe).__name__}: {_qe})")
                        print("  [DICOM] NOTE: quote the VESSEL numbers as the "
                              "headline result. The whole-mask numbers above are\n"
                              "           diluted by noise-dominated parenchyma "
                              "and understate the correction.")
                except Exception as _exc:
                    print(f"  [DICOM] Vessel-restricted statistics failed ({_exc}).")

                # Backwards-compat key for old summary code
                stats["mean_cm_s"]   = stats["delta_mean_cm_s"]
                stats["median_cm_s"] = stats["delta_median_cm_s"]
                stats["p95_cm_s"]    = stats["delta_p95_cm_s"]
                stats["max_cm_s"]    = stats["delta_max_cm_s"]

                # ═ ARTEFACTS ARE INDEPENDENT ═════════════════════════
                #
                # Each output below is written under its OWN handler. They
                # used to run unguarded, one after another, inside the single
                # Stage-5 try: the .npz save raised, and the exception flew
                # past the figure call AND past the dicom_summary assembly to
                # the outer handler — so a failure in a convenience artefact
                # cost the user fig05, the whole dicom_correction block in
                # run_config.json and the VTK export, and the run ended
                # claiming "DICOM CORRECTION: not run this session" even
                # though the correction had completed and printed its results.
                #
                # The correction result is the deliverable; the .npz is a
                # convenience. Never let the second destroy the first.

                # ─ Save corrected volume (optional, large) ───────────────
                _npz_name = None
                _npz_bytes = 2 * v_dcm.size * 4          # two float32 copies
                if not DICOM_SAVE_NPZ:
                    print("  [DICOM] Corrected volume NOT saved "
                          "(DICOM_SAVE_NPZ = False).")
                elif _npz_bytes > DICOM_NPZ_MAX_BYTES:
                    print(f"  [DICOM] Corrected volume NOT saved: it would "
                          f"need ~{_npz_bytes/2**30:.1f} GiB in memory before "
                          "compression,")
                    print(f"          over the {DICOM_NPZ_MAX_BYTES/2**30:.1f} "
                          "GiB limit (DICOM_NPZ_MAX_BYTES). The VTK export "
                          "below carries")
                    print("          the corrected field for ParaView; raise "
                          "the limit if you need the .npz.")
                else:
                    try:
                        out_path = Path(output_dir) / "dicom_corrected.npz"
                        np.savez_compressed(
                            out_path,
                            # astype() copies unconditionally; on already-
                            # float32 data that is two needless 556 MiB
                            # allocations at the point of writing.
                            v_measured=(v_dcm if v_dcm.dtype == np.float32
                                        else v_dcm.astype(np.float32)),
                            v_corrected=(v_corr_d
                                         if v_corr_d.dtype == np.float32
                                         else v_corr_d.astype(np.float32)),
                            inside_r0_mask=inside_r0_d,
                            pixel_spacing_mm=np.array([dx*1000, dy*1000, dz*1000]),
                            VENC_m_s=VENC_d,
                            B0_T=cfg.B0,
                            coil_name=str(grad_coeffs.coil_name),
                        )
                        _npz_name = out_path.name
                        print(f"  [DICOM] Corrected volume saved → {_npz_name}")
                    except Exception as _ne:
                        print(f"  [DICOM] Corrected volume NOT saved "
                              f"({type(_ne).__name__}: {_ne}).")
                        print("          Continuing — the figure, the JSON "
                              "summary and the VTK do not depend on it.")

                # ─ Diagnostic figure ─────────────────────────────────────
                try:
                    try:
                        plotter.plot_dicom_correction_summary(
                            v_dcm, v_corr_d, inside_r0_d,
                            coil=grad_coeffs.coil_name,
                        )
                        print("  [DICOM] Diagnostic figure saved → "
                              "fig05_dicom_correction.png")
                    except AttributeError:
                        # PlottingEngine has no DICOM helper — this is the
                        # NORMAL path, not an error. Note that an exception
                        # raised inside this handler propagates out of the
                        # whole try/except pair, which is why the outer guard
                        # below exists.
                        _dicom_quick_plot(v_dcm, v_corr_d, inside_r0_d,
                                           output_dir, coil=grad_coeffs.coil_name,
                                           lambda_dev=lambda_dev_d,
                                           Xd=Xd, Yd=Yd, Zd=Zd,
                                           mag_vol=mag_for_plot, venc=VENC_d,
                                           vessel_thr=_vthr_for_plot,
                                           vessel_sigma=_sigma_for_plot,
                                           flow_gate=_flow_gate,
                                           vessel_stats=vessel_stats,
                                           fig05_slice_override=getattr(gui, "fig05_slice", None))
                except Exception as _fe:
                    print(f"  [DICOM] fig05 FAILED "
                          f"({type(_fe).__name__}: {_fe}) — the correction "
                          "itself is unaffected;")
                    print("          the statistics below and in "
                          "run_config.json are still valid.")
                    traceback.print_exc(limit=6)

                dicom_summary = {
                    "layout_frac_over_venc": _layout_frac_over_venc,
                    "n_files":      len(dcm_files),
                    "volume_shape": list(v_dcm.shape),
                    "VENC_m_s":     VENC_d,
                    # ND / DIS2D / DIS3D token from ImageType — records
                    # whether the Λ position assumption (vendor-unwarped
                    # coordinates) was verified for this dataset.
                    # None is NOT "fine": it means the tag was absent (common
                    # after a MeVisLab / research re-export, which rewrites
                    # ImageType) and the assumption behind compute_lambda is
                    # therefore UNVERIFIED for this series. See the note key.
                    "geometry_correction": _geom_state,
                    # Verified isocentre precondition (Siemens 0019,1014).
                    "rel_table_position_mm": locals().get("_tbl_rel", None),
                    "geometry_correction_note": (
                        "UNVERIFIED but BOUNDED. ImageType carried no "
                        "ND/DIS2D/DIS3D token (common after a research "
                        "re-export), so it is unknown whether vendor "
                        "geometric distortion correction was applied. Markl "
                        "2003 Eq. [2] requires the TRUE unwarped position. "
                        "Evaluating Lambda at the displaced coordinates "
                        "instead bounds the induced error at roughly 2% of "
                        "the correction magnitude (median) and under 10% "
                        "worst case, EVEN under the worst assumption that no "
                        "geometric correction was applied at all — on a "
                        "head-sized slab about 0.03 cm/s, against a "
                        "per-voxel velocity noise floor two orders of "
                        "magnitude larger. Report it as a bounded limitation "
                        "rather than resolving the tag."
                        if _geom_state is None else
                        f"Distortion state {_geom_state}. Token read from "
                        f"ImageType (0008,0008) and/or the Siemens private "
                        f"ImageTypeText (0051,1016); a research re-export "
                        f"commonly rewrites ImageType and leaves the token "
                        f"only in the private tag, so cite the tag that "
                        f"actually carries it for this series."),
                    # M6 — what was actually DONE to the component axes, not
                    # what the module-level flags happen to be set to.
                    "axis_flags": _axis_applied,
                    "n_phases":     Nt_d,
                    "pixel_spacing_mm": [dx*1000, dy*1000, dz*1000],
                    "inside_r0_fraction": r0_only_frac,
                    "signal_fraction":    signal_only_frac,
                    "vessel_restricted":  vessel_stats,
                    "correction_magnitude_cm_s": stats,
                    # None when the volume was skipped or failed to save —
                    # the JSON must not claim a file that is not on disk.
                    "saved_volume": _npz_name,
                }
                if _mask_recovery is not None:
                    dicom_summary["mask_auto_recovery"] = _mask_recovery
                if _axis_autoassign is not None:
                    dicom_summary["axis_auto_assignment"] = _axis_autoassign
                if _mask_record is not None:
                    dicom_summary["signal_mask"] = _mask_record
                if _pipe_extra is not None:
                    dicom_summary["pipe_main_object_only"] = _pipe_extra

                # ── VTK export for 3D visualisation in ParaView ───────────
                # Writes the measured and corrected velocity fields (and the
                # anatomical magnitude) to a .vtk file. Open it in ParaView to
                # make streamlines / vector glyphs of the flow — the standard
                # 3D visualisation for 4D-flow, and what the diagnostic PNGs
                # (flat 2D slices) cannot show.
                try:
                    _origin_mm = (float(Xd.min()) * 1000.0,
                                  float(Yd.min()) * 1000.0,
                                  float(Zd.min()) * 1000.0)
                    # output_dir is a plain str, so wrap it in Path before
                    # using the / operator.
                    _vtk_path = Path(output_dir) / "velocity_field.vtk"
                    # Per-voxel temporal coherence and time-averaged speed,
                    # exported alongside the single displayed phase.
                    _vtk_coh = _vtk_tavg = None
                    if v_dcm.ndim == 5 and v_dcm.shape[-1] >= 4:
                        try:
                            _sp_all  = np.linalg.norm(v_dcm, axis=0)
                            _vtk_tavg = _sp_all.mean(axis=-1) * 100.0
                            _vtk_coh  = (
                                np.sqrt((v_dcm.mean(axis=-1) ** 2).sum(axis=0))
                                / np.maximum(_sp_all.mean(axis=-1), 1e-12))
                            print("  [vtk] Adding flow_coherence and "
                                  "speed_timeavg (over all "
                                  f"{v_dcm.shape[-1]} phases) — threshold on "
                                  "flow_coherence in ParaView to keep slow "
                                  "vessels without keeping tissue.")
                        except Exception as _cve:
                            print(f"  [vtk] coherence array skipped: {_cve}")
                    # The VTK image has axes x, y, z and an origin at the
                    # MINIMUM coordinate, so the volume must be handed over in
                    # patient axis order with every axis increasing along its
                    # patient axis. For an axial stack that is what it already
                    # is. For a sagittal/coronal one the array axes are a
                    # permutation (and some run backwards), and writing them
                    # as they stand shows the pipe along the wrong axis and
                    # mirrored, with the (patient-frame) velocity vectors not
                    # lining up with the grid.
                    _vtk_v, _vtk_vc, _vtk_mag = v_dcm, v_corr_d, mag_for_plot
                    _vtk_sp = (dx * 1000.0, dy * 1000.0, dz * 1000.0)
                    _pm_sg = array_axes_to_patient(locals().get("_iop", None),
                                                   return_signs=True)
                    if _pm_sg is None:
                        print("  [vtk] WARNING: the stack is not axis-aligned, so the "
                              "volume is written in array order and will NOT be "
                              "oriented correctly in ParaView.")
                    elif _pm_sg[0] != [0, 1, 2] or any(t < 0 for t in _pm_sg[1]):
                        _pm, _sg = _pm_sg
                        _tr = [int(i) for i in np.argsort(_pm)]
                        _vtk_v = reorient_to_patient(v_dcm, 1, _pm, _sg)
                        _vtk_vc = reorient_to_patient(v_corr_d, 1, _pm, _sg)
                        if mag_for_plot is not None:
                            _vtk_mag = reorient_to_patient(mag_for_plot, 0, _pm, _sg)
                        if _vtk_coh is not None:
                            _vtk_coh = reorient_to_patient(_vtk_coh, 0, _pm, _sg)
                        if _vtk_tavg is not None:
                            _vtk_tavg = reorient_to_patient(_vtk_tavg, 0, _pm, _sg)
                        _vtk_sp = tuple(_vtk_sp[a] for a in _tr)
                        print("  [vtk] Volume re-ordered to patient axes (x, y, z) "
                              "for export (array axes ran along patient ("
                              + ", ".join("xyz"[q] for q in _pm) + ")).")
                    _vtk_out = write_velocity_vtk(
                        str(_vtk_path),
                        v_meas=_vtk_v,
                        v_corr=_vtk_vc,
                        spacing_mm=_vtk_sp,
                        origin_mm=_origin_mm,
                        mag=_vtk_mag,
                        venc_cm_s=VENC_d * 100.0,
                        t_index=(None if DICOM_VTK_PHASES == "peak"
                                 else DICOM_VTK_PHASES),
                        # lean applies to the TIME SERIES only; a
                        # single-phase export always keeps anatomy and scalars.
                        lean=(DICOM_VTK_LEAN
                              if str(DICOM_VTK_PHASES).lower() == "all"
                              else False),
                        # Computed over every cardiac phase, so ParaView can
                        # separate slow FLOW from stationary tissue without a
                        # speed threshold (see the writer docstring).
                        coherence=_vtk_coh,
                        speed_timeavg_cm_s=_vtk_tavg)
                    # "all" returns a LIST of paths and does not create
                    # velocity_field.vtk itself, so measure what was returned.
                    _written = ([_vtk_out] if isinstance(_vtk_out, str)
                                else list(_vtk_out))
                    _vtk_mb = sum(Path(p).stat().st_size
                                  for p in _written) / 1024.0**2
                    if len(_written) == 1:
                        print(f"  [DICOM] 3D velocity field written → "
                              f"velocity_field.vtk  ({_vtk_mb:.0f} MB; open in "
                              f"ParaView: add a Glyph or Stream Tracer on "
                              f"'velocity_corrected').")
                        print("           NOTE: the exported field is "
                              "UNMASKED — air voxels are pure random-phase "
                              "noise averaging ~VENC. Threshold on "
                              "'anatomy_magnitude' before quoting any "
                              "statistic from this file.")
                    else:
                        print(f"  [DICOM] 3D velocity time series written → "
                              f"{len(_written)} files, {_vtk_mb/1024:.1f} GB "
                              f"total, in {Path(output_dir).name}")

                    # ── isocentre marker for ParaView ───────────────────
                    # Lambda = I at (0,0,0) by construction; the figure only
                    # makes its argument if that point is visible in it.
                    try:
                        _iso_p = Path(output_dir) / "isocentre.vtk"
                        write_isocentre_marker(str(_iso_p))
                        print("  [DICOM] Isocentre marker written → "
                              "isocentre.vtk  (load in ParaView beside "
                              "velocity_field.vtk; solid colour, line "
                              "width 3)")
                    except Exception as _exc:
                        print(f"  [DICOM] Isocentre marker skipped "
                              f"({type(_exc).__name__}: {_exc})")
                    dicom_summary["saved_vtk"] = [Path(p).name
                                                  for p in _written]

                    # ── ParaView threshold values for THIS dataset ──────────
                    # These are derived from the data, not conventions, so
                    # they change with every acquisition. Printing them (and
                    # storing them in run_config.json) means the numbers to
                    # type into ParaView never have to be looked up or
                    # remembered — and they stay reproducible for the thesis.
                    try:
                        _m = np.asarray(mag_for_plot, dtype=np.float32)
                        _mmax = float(_m.max())
                        _alo = 0.10 * _mmax        # same signal mask as above
                        _sthr = (float(vessel_stats["vessel_threshold_cm_s"])
                                 if vessel_stats else 17.0)
                        # When the velocity gate is deliberately disabled
                        # (DICOM_VESSEL_THRESHOLD_CMS = 0, the in-vivo preset)
                        # a "0 to max" hint tells the reader nothing, so fall
                        # back to the 17 cm/s default for DISPLAY purposes
                        # only. This does not affect any statistic.
                        if not np.isfinite(_sthr) or _sthr <= 0:
                            _sthr = 17.0
                        # Peak over the cardiac cycle, computed FRAME BY FRAME.
                        # The previous version evaluated
                        #     np.linalg.norm(v_corr_d - v_dcm, axis=0)
                        # on the whole 5-D volume, which materialises a full
                        # (3, Nx, Ny, Nz, Nt) difference plus two 4-D norms —
                        # about 2.4 GB transiently at 260x320x52x17. On a
                        # machine without that headroom it raised MemoryError,
                        # the except below swallowed it, and run_config.json
                        # silently lost the paraview_thresholds block. Peak
                        # over t is the right basis for an UPPER display
                        # threshold and needs only one frame at a time.
                        _sp_c = _cm = None
                        if v_corr_d.ndim == inside_r0_d.ndim + 2:
                            for _t in range(v_corr_d.shape[-1]):
                                _s1 = np.linalg.norm(
                                    v_corr_d[..., _t], axis=0) * 100.0
                                _c1 = np.linalg.norm(
                                    v_corr_d[..., _t] - v_dcm[..., _t],
                                    axis=0) * 100.0
                                _sp_c = _s1 if _sp_c is None else np.maximum(_sp_c, _s1)
                                _cm = _c1 if _cm is None else np.maximum(_cm, _c1)
                                del _s1, _c1
                        else:
                            _sp_c = np.linalg.norm(v_corr_d, axis=0) * 100.0
                            _cm = np.linalg.norm(v_corr_d - v_dcm, axis=0) * 100.0
                        # Display max from the 95th percentile ON VESSELS, not
                        # from the global maximum. A handful of edge voxels can
                        # be 10x the typical correction, and scaling to them
                        # leaves the real structure squashed into the bottom of
                        # the colour bar — the map then looks uniformly dark
                        # and shows nothing.
                        # Vessels = fast AND inside the object. Without the
                        # signal mask this picks up background air, where the
                        # phase is random and the "correction" is large, and
                        # the p95 comes out several times too high.
                        _ves_m = inside_r0_d & (_sp_c > _sthr)
                        _pool = _cm[_ves_m] if _ves_m.sum() > 100 else \
                                _cm[inside_r0_d]
                        _cdisp = float(np.percentile(_pool, 95))
                        _cdisp = max(round(_cdisp * 2) / 2.0, 0.5)  # 0.5 steps
                        _smax = float(_sp_c.max())
                        _pv = {
                            "anatomy_magnitude_lower": round(_alo, 1),
                            "anatomy_magnitude_upper": round(_mmax, 1),
                            "speed_corrected_lower_cm_s": round(_sthr, 1),
                            "speed_corrected_upper_cm_s": round(_smax, 1),
                            "correction_mag_display_max_cm_s": _cdisp,
                        }
                        dicom_summary["paraview_thresholds"] = _pv
                        print("\n  ── ParaView values for THIS dataset "
                              "(also in run_config.json) ──")
                        print(f"     Threshold 1  anatomy_magnitude   "
                              f"{_pv['anatomy_magnitude_lower']:.0f}  to  "
                              f"{_pv['anatomy_magnitude_upper']:.0f}"
                              "        (keeps the head)")
                        print(f"     Threshold 2  speed_corrected     "
                              f"{_pv['speed_corrected_lower_cm_s']:.0f}  to  "
                              f"{_pv['speed_corrected_upper_cm_s']:.0f} cm/s  "
                              "  (keeps the vessels)")
                        print(f"     Colour map   correction_mag      0  to  "
                              f"{_pv['correction_mag_display_max_cm_s']:g}"
                              " cm/s    (p95 on vessels; the correction figure)")
                    except Exception as _pe:
                        print(f"  [DICOM] ParaView hint values unavailable "
                              f"({type(_pe).__name__}: {_pe})")
                except Exception as _ve:
                    # Report the exception type and a traceback so a failed
                    # export is not mistaken for an intentionally skipped one.
                    print(f"  [DICOM] VTK export FAILED "
                          f"({type(_ve).__name__}): {_ve}")
                    traceback.print_exc(limit=3)
        except Exception as e:
            # limit=2 hid the frame that actually raised, which made this
            # failure mode ("4 figures, DICOM not run") undiagnosable from a
            # pasted console log. Print enough of the stack to locate it.
            print(f"  [DICOM] STAGE 5 FAILED ({type(e).__name__}): {e}")
            print("          The correction results printed above may be "
                  "complete, but fig05 and the")
            print("          dicom_correction block of run_config.json were "
                  "NOT written. Traceback:")
            traceback.print_exc(limit=12)

    # ── Append final results to config log (with all metrics & SNR sweep) ──
    config_log["final_results"] = {}
    for mode, res in all_results.items():
        config_log["final_results"][f"mode_{mode}"] = {
            # Headline: vessel region (|v_true| > VESSEL_THRESH) — matches figures
            "rmse_before_cm_s":   float(res.rmse_before_vessel * 100),
            "rmse_after_cm_s":    float(res.rmse_after_vessel  * 100),
            "rmse_before_ci_cm_s": [float(c * 100) for c in res.rmse_before_ci],
            "rmse_after_ci_cm_s":  [float(c * 100) for c in res.rmse_after_ci],
            "mae_before_cm_s":    float(res.mae_before_vessel  * 100),
            "mae_after_cm_s":     float(res.mae_after_vessel   * 100),
            # Per-component correlations are the ones to quote; r_mag is
            # scale-invariant and near 1 regardless of the correction.
            # Retained for reproducibility only — NOT a reported result.
            # null = NaN (undefined, not zero): a reference component with no
            # spread across the ROI has no correlation to measure.
            "pearson_diagnostics_not_reported": {
                "why_not_reported": (
                    "Pearson r is invariant under linear scaling; gradient "
                    "nonlinearity is a multiplicative distortion, so r is "
                    "near-saturated both before and after correction and "
                    "carries no information about it. Quote RMSE / MAE / p95 "
                    "instead."),
                "before_xyz_sphere": _json_safe(res.pearson_before[:3]),
                "after_xyz_sphere":  _json_safe(res.pearson_after[:3]),
                "before_xyz_vessel": _json_safe(res.pearson_before_vessel[:3]),
                "after_xyz_vessel":  _json_safe(res.pearson_after_vessel[:3]),
            },
            "pct_improvement":    float(res.pct_improvement),
            "pct_improvement_ci": [float(c) for c in res.pct_improvement_ci],
            "pct_improvement_ci_method":
                "paired block bootstrap (x-slice blocks, 1000 replicates, "
                "percentile CI of the within-replicate improvement)",
            "metric_region":      "vessel (|v_true|>5 cm/s, inside R0)",
            "metric_definition":  "vector-error RMSE ||v_est-v_true||",
            # ── The same comparison, three ways (see ValidationResults) ──
            # The block above is computed on the TIME-AVERAGED field, which
            # suppresses noise by sqrt(Nt) but not the systematic GNL error,
            # so its percentage depends strongly on snr_db. Two SNR-honest
            # alternatives are recorded next to it so no figure can be quoted
            # without its noise assumption.
            "metric_basis": ("time-averaged field; see per_frame and "
                             "systematic_noise_free for the alternatives"),
            "snr_db": (None if not np.isfinite(res.snr_db)
                       else float(res.snr_db)),
            "per_frame": {
                "rmse_before_cm_s": float(res.rmse_before_vessel_perframe*100),
                "rmse_after_cm_s":  float(res.rmse_after_vessel_perframe*100),
                "pct_improvement":  float(res.pct_improvement_perframe),
                "note": "no temporal averaging; noise not suppressed",
            },
            "systematic_noise_free": {
                "rmse_before_cm_s": float(res.rmse_systematic_before_vessel*100),
                "rmse_after_cm_s":  float(res.rmse_systematic_after_vessel*100),
                "pct_improvement":  float(res.pct_improvement_systematic),
                "note": ("run repeated with noise OFF: the pure systematic "
                         "GNL error and the pure residual. SNR-independent. "
                         "THIS is the number that characterises the "
                         "correction. It is near-total by construction "
                         "because the forward model and the inverse use the "
                         "same Lambda -- it verifies the inversion, it does "
                         "not validate the field model (see self_tests.T5)."),
            },
            "discretisation": {
                "v_peak_configured_cm_s": float(res.v_peak_configured_m_s*100),
                "v_peak_on_grid_cm_s":    float(res.v_peak_on_grid_m_s*100),
                "peak_shortfall_pct": (
                    float(100.0*(1.0 - res.v_peak_on_grid_m_s /
                                 max(res.v_peak_configured_m_s, 1e-12)))),
                "voxels_per_feature": float(res.voxels_per_fwhm),
                "note": ("all % errors are referenced to the on-grid peak, "
                         "which is partial-volumed below the configured "
                         "value on a coarse grid"),
            },
            # Whole-sphere diagnostics (dominated by edge noise amplification)
            "rmse_before_whole_sphere_cm_s": float(res.rmse_before_whole * 100),
            "rmse_after_whole_sphere_cm_s":  float(res.rmse_after_whole  * 100),
            "rmse_before_ci_whole_cm_s": [float(c*100) for c in res.rmse_before_ci_whole],
            "rmse_after_ci_whole_cm_s":  [float(c*100) for c in res.rmse_after_ci_whole],
            "pct_improvement_whole_sphere":  float(res.pct_improvement_whole),
            "flow_rate_ref_ml_min":  float(res.flow_rate_ref),
            "flow_rate_std_ml_min":  float(res.flow_rate_std),
            "flow_rate_cor_ml_min":  float(res.flow_rate_cor),
        }
    if _snr_rows:
        config_log["snr_sensitivity"] = {
            "why": ("the percentage improvement of a synthetic GNL experiment "
                    "is set by the assumed SNR, because the residual after "
                    "correction is essentially pure noise. Quote this table, "
                    "or the systematic_noise_free row, not a single %."),
            "mode": int(list(all_results.keys())[-1]),
            "rows": _snr_rows,
        }
    # DICOM correction results (when DICOM pipeline was run, )
    if dicom_summary is not None:
        config_log["dicom_correction"] = dicom_summary
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(config_log, f, indent=2, default=str)
    print(f"\n  → Final results appended to: {cfg_path.name}")

    # ── Final summary ─────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("  SIMULATION COMPLETE")
    print(f"{'='*65}")
    print(f"  Coil:         {grad_coeffs.coil_name}")
    print(f"  R₀:           {grad_coeffs.R0*1000:.0f} mm")
    print(f"  N_max:        {grad_coeffs.max_degree}")
    print(f"  Grid:         {N}³  ({N**3} voxels)  ±{r_max*1000:.0f} mm")
    _mode_desc  = {1: ("mildly pulsatile Poiseuille tube (legacy)"
                       if getattr(cfg, "mode1_pulsatile", False)
                       else "steady Poiseuille tube"),
                   2: "pulsatile vessel",
                   3: "steady + pulsatile"}.get(sim_mode, "")
    _phase_word = ("phases (steady)" if (sim_mode == 1 and not getattr(
        cfg, "mode1_pulsatile", False)) else "cardiac phases")
    print(f"  Nt:           {cfg.Nt} {_phase_word}")
    print(f"  Flow:         {_mode_desc}  (pattern: {cfg.flow_pattern})")
    print()
    print("  ── SIMULATION RESULTS (vs synthetic ground truth) ──────────")
    print("  NOTE: These RMSE values compare v_corrected against the")
    print("  SYNTHETIC velocity field the program invented internally.")
    print("  They do NOT describe your DICOM data. See DICOM section below.")
    print()
    for mode, res in all_results.items():
        ml = "Poiseuille" if mode == 1 else "Pulsatile vessel"
        _vlab = getattr(res, "vessel_thresh_label", None)
        _vtxt = (f"time-avg |v_true|>{_vlab}" if _vlab
                 else "vessel-masked")
        print(f"  Mode {mode} ({ml})  [vessel region, {_vtxt}]:")
        print(f"    RMSE vs synthetic truth:  {res.rmse_before_vessel*100:.3f} → "
              f"{res.rmse_after_vessel*100:.3f} cm/s  "
              f"(improvement: {res.pct_improvement:.1f}%)")
        print(f"    RMSE 95%CI:  before [{res.rmse_before_ci[0]*100:.3f}, "
              f"{res.rmse_before_ci[1]*100:.3f}] →  "
              f"after [{res.rmse_after_ci[0]*100:.3f}, "
              f"{res.rmse_after_ci[1]*100:.3f}] cm/s")
    print()
    if dicom_summary is not None:
        s = dicom_summary["correction_magnitude_cm_s"]
        print("  ── DICOM CORRECTION RESULTS (your real data) ───────────────")
        print("  NOTE: No ground truth for acquired data. Results show how")
        print("  much the correction CHANGED the velocities (not error vs truth).")
        print(f"  DICOM volume: {dicom_summary['volume_shape']}  "
              f"VENC={dicom_summary['VENC_m_s']*100:.0f} cm/s")
        print(f"  Voxels inside R₀: {dicom_summary['inside_r0_fraction']*100:.0f}%")
        if dicom_summary.get("signal_fraction") is not None:
            print("  Voxels with signal (analysis set): "
                  f"{dicom_summary['signal_fraction']*100:.0f}%")
        print("  Velocity magnitude (mean, inside R₀, time-avg):")
        print("    Before correction: "
              f"{s.get('vmeas_mean_cm_s', s.get('mean_cm_s',0)):.2f} cm/s")
        print("    After  correction: "
              f"{s.get('vcorr_mean_cm_s', s.get('mean_cm_s',0)):.2f} cm/s")
        _vs = dicom_summary.get("vessel_restricted") if dicom_summary else None
        if _vs:
            if _vs.get("velocity_gate_disabled"):
                print("  ── Analysis set: magnitude/brain mask, no velocity "
                      f"gate (n = {_vs['n_vessel_voxels']:,}) ──")
                _ncs = _vs.get("noise_consistency") or {}
                if _ncs.get("frac_at_or_below_noise") is not None:
                    print(f"    {100*_ncs['frac_at_or_below_noise']:.0f}% of "
                          "this set is at or below the pure-noise speed "
                          f"({_ncs.get('noise_mean_speed_cm_s',0):.1f} cm/s):")
                    print("      the RELATIVE % below is scale-free and stays "
                          "valid; the ABSOLUTE cm/s is")
                    print("      diluted by stationary tissue and is not a "
                          "blood-flow number.")
            else:
                print(f"  ── Vessel voxels only (|v| > "
                      f"{_vs['vessel_threshold_cm_s']:.0f} cm/s, "
                      f"n = {_vs['n_vessel_voxels']:,}) ──")
            print(f"    Mean |v| over the set: {_vs['vmeas_mean_cm_s']:.1f} cm/s")
            print(f"    Correction: mean {_vs['delta_mean_cm_s']:.3f}  "
                  f"median {_vs['delta_median_cm_s']:.3f}  "
                  f"p95 {_vs['delta_p95_cm_s']:.3f} cm/s")
            print(f"    (mean relative correction: "
                  f"{_vs['pct_signal_corrected']:.1f}% = mean|Δv|/mean|v|)")
        print("  Correction magnitude |v_cor − v_meas|:")
        print(f"    median = {s['median_cm_s']:.3f} cm/s  "
              f"  p95 = {s['p95_cm_s']:.3f} cm/s")
        print("  → See fig05_dicom_correction.png for spatial maps")
    else:
        print("  ── DICOM CORRECTION: not run this session ──────────────────")
        print("  (Select a DICOM folder in the GUI to run correction on real data)")
    print()
    print(f"  Output figures: {output_dir}")
    n_dicom_fig = 1 if dicom_summary is not None else 0
    print(f"  Total figures:  {4 + n_dicom_fig} PNG files")
    print("  Run config:     run_config.json  (full reproducibility)")
    print(f"  B0 (scanner):   {cfg.B0:.1f} T  ·  {scanner_tag}")
    print(f"{'='*65}\n")

    # Keep the console open when launched by double-click on Windows, but
    # never block a redirected/batch run.
    if sys.stdin is not None and sys.stdin.isatty():
        input("  Press Enter to exit …")


def _write_aborted_record(exc: BaseException) -> None:
    """
    Append an `aborted` section to run_config.json when a run does not finish.

    A reproducibility record has to record failures too. Previously a crash
    anywhere between the early write and the final write left run_config.json
    ending after `self_tests`, indistinguishable from a run that simply had no
    DICOM stage — and with the console gone there was nothing left to diagnose
    from. Note that BaseException is caught deliberately: KeyboardInterrupt and
    MemoryError are among the likeliest ways a long run on a large 4D volume
    ends, and neither is an `Exception`.
    """
    path, log = _RUN_STATE.get("path"), _RUN_STATE.get("log")
    if path is None or log is None:
        return
    try:
        log["aborted"] = {
            "completed": False,
            "why": ("the run did not reach the final write, so final_results, "
                    "snr_sensitivity and dicom_correction are ABSENT and no "
                    "number in this file should be quoted as a completed run"),
            "exception_type": type(exc).__name__,
            "exception_message": str(exc)[:2000],
            "traceback": traceback.format_exc()[-8000:],
            "aborted_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(log, f, indent=2, default=str)
        print(f"\n  [abort] run_config.json updated with the failure record "
              f"({type(exc).__name__}). The run did NOT complete.")
    except Exception as _e:            # never mask the original failure
        print(f"  [abort] could not write the failure record: {_e}")


if __name__ == "__main__":
    try:
        main()
    except BaseException as _exc:
        _write_aborted_record(_exc)
        raise