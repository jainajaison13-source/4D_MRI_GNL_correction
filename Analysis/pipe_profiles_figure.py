#!/usr/bin/env python3
"""
pipe_profiles_figure.py — Fig. 4.15 of the thesis: (a) relative correction
sum|dv| / sum|v_meas| per cross-sectional plane and (b) plane-mean |dv| along
the tube, for the four pipe-flow series, from the fields exported by the toolbox.

Each series folder must hold run_config.json and velocity_field.vtk. The lumen
is rebuilt as the toolbox built it (magnitude >= the recorded signal threshold,
largest connected object) and checked against the recorded voxel count. The
tube axis is the array axis with the largest lumen extent; the position x of a
plane comes from the VTK origin and spacing. Planes whose lumen cross-section
deviates by more than 15 % from the median are omitted (the rule of Tab. 4.5).

Usage:
    python pipe_profiles_figure.py <0081 folder> <0080 folder> <0089 folder> <0088 folder> [out.png]
Prints the mean ND - DIS2D difference of the relative correction on the planes
both series cover, and the flow-rate change on the common segment x = -62..+22 mm.
"""
import sys, os, json
import numpy as np
from scipy import ndimage
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from readvtk import read as read_vtk

plt.rcParams["font.family"] = "DejaVu Sans"
SERIES = [("1 mm DIS2D (0081)", "#1f4e9c", "-"), ("1 mm ND (0080)", "#6fa8dc", "--"),
          ("0.67 mm DIS2D (0089)", "#a61c1c", "-"), ("0.67 mm ND (0088)", "#f0a030", "--")]
if len(sys.argv) < 5:
    sys.exit(__doc__)
out = sys.argv[5] if len(sys.argv) > 5 else "pipe_profiles_along_tube.png"

curves = {}
for folder, (label, col, ls) in zip(sys.argv[1:5], SERIES):
    rec = json.load(open(os.path.join(folder, "run_config.json"), encoding="utf-8"))
    dc = rec["dicom_correction"]; vr = dc["vessel_restricted"]
    hdr, d = read_vtk(os.path.join(folder, "velocity_field.vtk"), {"anatomy_magnitude", "correction_mag", "speed_measured", "velocity_measured", "velocity_corrected"})
    mag = d["anatomy_magnitude"]
    thr = dc["signal_mask"]["threshold"]
    lab, _ = ndimage.label(mag >= thr); sizes = np.bincount(lab.ravel())[1:]; lumen = lab == (np.argmax(sizes) + 1)
    n_rec = vr["n_vessel_voxels"]
    print(f"{label:22s} lumen voxels {lumen.sum()} (run record {n_rec}){'' if lumen.sum() == n_rec else '  <-- MISMATCH'}")
    if "correction_mag" in d and "speed_measured" in d:
        dabs = d["correction_mag"]; sm = d["speed_measured"]
    else:
        vm = d["velocity_measured"].astype(np.float64); vc = d["velocity_corrected"].astype(np.float64)
        dabs = np.sqrt(((vc - vm) ** 2).sum(0)); sm = np.sqrt((vm ** 2).sum(0))
    ext = [np.ptp(np.nonzero(lumen)[k]) for k in range(3)]; ax_t = int(np.argmax(ext))
    other = tuple(k for k in range(3) if k != ax_t)
    area = lumen.sum(axis=other); planes = np.nonzero(area)[0]
    x = hdr["origin"][ax_t] + planes * hdr["spacing"][ax_t]
    fc = vr["flux_conservation"][0]
    if fc.get("lumen_area_vox"):
        # validity exactly as in Tab. 4.5: per-plane lumen area of the flux test in the run
        # record, within 15 % of its median. The record counts planes from the far end of
        # the tube, so record plane i is VTK plane (n-1-i).
        n_pl = mag.shape[ax_t]; ar = np.array(fc["lumen_area_vox"], float); idx = np.array(fc["plane_index"])
        ok_rec = np.abs(ar / np.median(ar) - 1) <= 0.15
        valid_k = set((n_pl - 1 - idx[ok_rec]).tolist()); valid = np.array([k in valid_k for k in planes])
        rule = "run-record lumen area"
    else:
        med = np.median(area[planes]); valid = np.abs(area[planes] / med - 1) <= 0.15; rule = "lumen area of the exported field"
    rel = np.array([100 * (dabs * lumen).take(p, axis=ax_t).sum() / max((sm * lumen).take(p, axis=ax_t).sum(), 1e-9) for p in planes])
    ab = np.array([(dabs * lumen).take(p, axis=ax_t).sum() / max(area[p], 1) for p in planes])
    curves[label] = (x[valid], rel[valid], ab[valid], col, ls)
    print(f"{'':22s} tube axis = array axis {ax_t}, {len(planes)} planes, {valid.sum()} valid ({rule}); x = {x.min():.1f} .. {x.max():.1f} mm; "
          f"relative correction {rel[valid].min():.2f} .. {rel[valid].max():.2f} %")
    # flow-rate change on the common segment, from the record's per-plane flow rates
    if fc.get("q_before_ml_min"):
        qb = np.abs(np.array(fc["q_before_ml_min"])); qa = np.abs(np.array(fc["q_after_ml_min"]))
        idx = np.array(fc["plane_index"]) if fc.get("plane_index") else np.arange(len(qb))
        if fc.get("lumen_area_vox"):
            ar = np.array(fc["lumen_area_vox"], float); okp = np.abs(ar / np.median(ar) - 1) <= 0.15
        else:
            okp = np.ones(len(qb), bool)
        # the record counts planes from the other end of the tube: x_i = x_max - i*dx
        xi = x.max() - idx * hdr["spacing"][ax_t]
        seg = okp & (xi >= -62) & (xi <= 22)
        print(f"{'':22s} flow-rate change on the common segment x = -62..+22 mm: {100 * (qa[seg].mean() / qb[seg].mean() - 1):+.2f} % ({seg.sum()} planes)")

for a, b in [("1 mm DIS2D (0081)", "1 mm ND (0080)"), ("0.67 mm DIS2D (0089)", "0.67 mm ND (0088)")]:
    if a in curves and b in curves:
        xa, ra = curves[a][0], curves[a][1]; xb, rb = curves[b][0], curves[b][1]
        lo, hi = max(xa.min(), xb.min()), min(xa.max(), xb.max()); xs = xa[(xa >= lo) & (xa <= hi)]
        diff = np.interp(xs, xb[np.argsort(xb)], rb[np.argsort(xb)]) - np.interp(xs, xa[np.argsort(xa)], ra[np.argsort(xa)])
        print(f"ND - DIS2D, relative correction per plane: mean {diff.mean():+.3f} percentage points, max |.| {np.abs(diff).max():.3f}  ({b} vs {a})")

fig, ax = plt.subplots(1, 2, figsize=(12, 4.3))
def gapped(x, y):
    # insert NaN where omitted planes interrupt the curve, so that gaps are not bridged
    o = np.argsort(x); x, y = x[o], y[o]; dx = np.median(np.diff(x)) if len(x) > 1 else 1.0
    xs, ys = [x[0]], [y[0]]
    for i in range(1, len(x)):
        if x[i] - x[i - 1] > 1.5 * dx:
            xs.append(np.nan); ys.append(np.nan)
        xs.append(x[i]); ys.append(y[i])
    return np.array(xs), np.array(ys)
for label, (x, rel, ab, col, ls) in curves.items():
    xg, rg = gapped(x, rel); _, ag = gapped(x, ab)
    ax[0].plot(xg, rg, color=col, ls=ls, lw=1.6, label=label); ax[1].plot(xg, ag, color=col, ls=ls, lw=1.6, label=label)
ax[0].set_title("(a) Relative correction  Σ|Δv| / Σ|v_meas|  per plane"); ax[0].set_ylabel("relative correction [%]")
ax[1].set_title("(b) Absolute correction, plane-mean |Δv|"); ax[1].set_ylabel("|Δv| [cm/s]")
for a_ in ax:
    a_.set_xlabel("Position along tube, x [mm]"); a_.legend(fontsize=9); a_.grid(alpha=0.3)
    a_.spines["top"].set_visible(False); a_.spines["right"].set_visible(False)
fig.tight_layout(); fig.savefig(out, dpi=220, bbox_inches="tight")
print("written", out)
