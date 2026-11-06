#!/usr/bin/env python3
"""
pipe_raw_flow_figure.py — flow rate per cross-section along the pipe for the
four pipe-flow series, before and after the gradient correction (thesis Fig. 4.16).

Reads the run records (run_config.json) of the four runs and plots the
per-plane flow rate before and after the gradient correction, which the toolbox
stores in dicom_correction.vessel_restricted.flux_conservation[0].q_before_ml_min
/ q_after_ml_min
with the plane index of every cross-section. Nothing is recomputed.

  (a) |Q| per plane in ml/min against the position along the tube, before the
      correction (solid/dashed) and after it (thin lines, q_after_ml_min)
  (b) the same normalised to the mean of each series over its valid planes
      (before correction)

Plane position: the flux test numbers the planes in the slice order of the
DICOM series, which for these acquisitions runs from the first slice at
x_first towards negative patient x (verified against the per-slice panel of
the toolbox summary and the exported VTK origin): x_i = x_first - i*dx.
  1 mm series   : x_first = +70.88 mm, dx = 1.0 mm   (VTK origin -88.12 mm)
  0.67 mm series: x_first = +44.6 mm,  dx = 0.67 mm  (DICOM header origin)
Valid planes: as in Tab. 4.5 of the thesis, planes whose segmented lumen
cross-section (lumen_area_vox in the run record) deviates by more than 15 %
from the median are shown pale and excluded from the normalisation; in the
0.67 mm series these are the distal planes (x > +25 mm), where the lumen
segmentation collapsed, one corrupted slice and the last plane.

Usage: python pipe_raw_flow_figure.py run81.json run80.json run89.json run88.json out.png
"""
import sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

f81, f80, f89, f88, out = sys.argv[1:6]
series = [  # (file, label, x_first, dx, colour, linestyle)
    (f81, "1 mm DIS2D (0081)",    70.88, 1.00, "#1f4e9c", "-"),
    (f80, "1 mm ND (0080)",       70.88, 1.00, "#6fa8dc", "--"),
    (f89, "0.67 mm DIS2D (0089)", 44.60, 0.67, "#a51c1c", "-"),
    (f88, "0.67 mm ND (0088)",    44.60, 0.67, "#e8912d", "--"),
]
X_VALID_MAX = 25.0   # mm; distal planes of the 0.67 mm series beyond this are invalid (Tab. 4.5)

plt.rcParams["font.family"] = "DejaVu Sans"   # matplotlib's bundled font: identical output on every machine
plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
                     "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 9.5})
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8.4))   # stacked: printed at text width the fonts stay >= 8 pt
means = {}
for path, label, x0, dx, col, ls in series:
    d = json.load(open(path, encoding="utf-8"))
    fc = d["dicom_correction"]["vessel_restricted"]["flux_conservation"][0]
    idx = np.array(fc["plane_index"]); q = -np.array(fc["q_before_ml_min"])   # flow is in -x; show magnitude
    qa = -np.array(fc["q_after_ml_min"])                                        # the same planes after the correction
    x = x0 - idx * dx
    if fc.get("lumen_area_vox") is not None:    # same rule as Tab. 4.5: lumen area within 15 % of the median
        area = np.array(fc["lumen_area_vox"], float)
        valid = np.abs(area / np.median(area) - 1) <= 0.15
    else:                                       # older run record without the per-plane area
        valid = x <= X_VALID_MAX if dx < 1 else np.ones_like(x, bool)
    m = q[valid].mean(); means[label] = (m, int(valid.sum()))
    o = np.argsort(x); xv, qv, vv = x[o], q[o], valid[o]
    qline = np.where(vv, qv, np.nan)    # valid planes as a line with gaps at the excluded planes
    ax1.plot(xv, qline, ls, color=col, lw=1.4, label=label)
    qav = qa[o]; ax1.plot(xv, np.where(vv, qav, np.nan), ls, color=col, lw=0.6, alpha=0.9)   # after correction: thin line, same colour
    if (~vv).any():                     # excluded planes: pale dots, not in the legend
        ax1.plot(xv[~vv], qv[~vv], "o", color=col, ms=2.5, alpha=0.35, mew=0)
    ax2.plot(xv, 100 * (qline / m - 1), ls, color=col, lw=1.4, label=label)
    ma = qa[valid].mean()
    print(f"{label:22s} planes {len(q):3d}, valid {int(valid.sum()):3d}, mean |Q| over valid planes {m:6.1f} ml/min before, {ma:6.1f} after ({100*(ma/m-1):+.2f} %)")
ax1.axhline(300, color="0.4", lw=0.8, ls=":")
ax1.text(48, 302, "nominal 300 ml/min", fontsize=9, color="0.3", va="bottom")
ax1.axvspan(X_VALID_MAX, 46, color="0.92", zorder=0)
ax1.text((X_VALID_MAX + 46) / 2, 352, "0.67 mm series:\nlumen collapsed\n(excluded, dots)", fontsize=8.5, color="0.35", ha="center", va="top")
ax1.set_xlabel("Position along tube, x [mm]"); ax1.set_ylabel("Measured flow rate |Q| [ml/min]")
ax1.set_title("(a) Flow rate per cross-section before (thick) and after (thin) correction")
ax1.set_ylim(80, 360)
ax1.grid(alpha=0.3); ax1.legend(loc="lower left", ncol=2)
ax2.axhline(0, color="0.4", lw=0.8, ls=":")
ax2.set_xlabel("Position along tube, x [mm]"); ax2.set_ylabel("Deviation from series mean [%]")
ax2.set_title("(b) Normalised to the mean of each series (valid planes)")
ax2.grid(alpha=0.3); ax2.legend(loc="lower left", ncol=2)
for ax in (ax1, ax2):
    ax.set_xlim(-92, 75)
fig.tight_layout()
fig.savefig(out, dpi=200)
print("ND / DIS2D ratio of the means: 1 mm %.3f, 0.67 mm %.3f" % (
    means["1 mm ND (0080)"][0] / means["1 mm DIS2D (0081)"][0],
    means["0.67 mm ND (0088)"][0] / means["0.67 mm DIS2D (0089)"][0]))
print("saved", out)
