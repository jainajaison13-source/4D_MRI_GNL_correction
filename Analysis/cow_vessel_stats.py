#!/usr/bin/env python3
"""
cow_vessel_stats.py — the in vivo numbers of the thesis that are evaluated on
the VESSEL SET (Sections 4.5.1–4.5.3, Tab. 4.8 vessel column, Tab. 4.9), from
the velocity field exported by the toolbox (velocity_field.vtk of the Circle of
Willis run). The analysis-set numbers are in run_config.json and are not
recomputed here.

Vessel set as in the thesis: toolbox brain mask (rebuilt with the toolbox's own
_build_brain_mask), anatomy magnitude >= the run record's paraview_thresholds/
anatomy_magnitude_lower (41.1 for the in vivo run), flow coherence >= 0.80.
The exported velocity vectors are the peak-systolic frame; speed_timeavg is the
time-averaged |v| over the 17 cardiac phases.

Prints:
  Section 4.5.1  vessel-set size, radii, coherence of the arterial tree (largest
                 connected component) and of the remaining brain voxels, slowest
                 retained flow, mean velocity vector of static tissue (offset)
  Table 4.8      vessel column
  Table 4.9      correction binned by distance from the isocentre
  Section 4.5.3  patchiness of the signed magnitude change

Usage: python cow_vessel_stats.py velocity_field.vtk [--sigma-v 4.25]
       (needs 4D_MRI_GNL_correction.py in the same or the parent folder for the brain mask)
"""
import sys, glob, os, argparse
import numpy as np
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from readvtk import read as read_vtk

ap = argparse.ArgumentParser()
ap.add_argument("vtk"); ap.add_argument("--sigma-v", type=float, default=4.25)
ap.add_argument("--mag-min", type=float, default=41.1); ap.add_argument("--coh-min", type=float, default=0.8)
a = ap.parse_args()

hdr, d = read_vtk(a.vtk, {"velocity_measured", "velocity_corrected", "anatomy_magnitude", "flow_coherence", "speed_timeavg"})
vm = d["velocity_measured"].astype(np.float64); vc = d["velocity_corrected"].astype(np.float64)
mag = d["anatomy_magnitude"]; coh = d["flow_coherence"]; st = d["speed_timeavg"]
_here = os.path.dirname(os.path.abspath(__file__))
tb = (sorted(glob.glob(os.path.join(_here, "4D_MRI_GNL_correction*.py")))
      or sorted(glob.glob(os.path.join(os.path.dirname(_here), "4D_MRI_GNL_correction*.py"))))
src = open(tb[0], encoding="utf-8").read(); i = src.index("def _build_brain_mask"); j = src.index("\ndef ", i + 10)
ns = {}; exec("import numpy as np\nfrom scipy import ndimage\n" + src[i:j], ns)
brain = ns["_build_brain_mask"](mag)
signal = brain & (mag >= a.mag_min)
vessel = signal & (coh >= a.coh_min)
tissue = signal & ~vessel

nx, ny, nz = mag.shape; o = hdr["origin"]; sp = hdr["spacing"]
X, Y, Z = np.meshgrid(o[0] + np.arange(nx) * sp[0], o[1] + np.arange(ny) * sp[1], o[2] + np.arange(nz) * sp[2], indexing="ij")
R = np.sqrt(X ** 2 + Y ** 2 + Z ** 2)
sm = np.sqrt((vm ** 2).sum(0)); sc = np.sqrt((vc ** 2).sum(0)); dv = vc - vm
dabs = np.sqrt((dv ** 2).sum(0)); rel = dabs / np.maximum(sm, 1e-9)

print("=== Section 4.5.1  vessel selection")
lab, _ = ndimage.label(vessel); sizes = np.bincount(lab.ravel())[1:]; tree = lab == (np.argmax(sizes) + 1)
print("vessel voxels: %d   (brain-mask signal voxels: %d)" % (vessel.sum(), signal.sum()))
print("distance from the isocentre: vessel set %.1f–%.1f mm; arterial tree (largest connected component, %d voxels) %.1f–%.1f mm"
      % (R[vessel].min(), R[vessel].max(), tree.sum(), R[tree].min(), R[tree].max()))
print("coherence median: arterial tree %.3f | remaining brain voxels %.3f" % (np.median(coh[tree]), np.median(coh[tissue])))
h, _ = np.histogram(coh[signal], bins=np.arange(0, 1.001, 0.1))
print("coherence histogram of the brain-mask voxels, bins of 0.1 from 0: " + " ".join("%d" % v for v in h) + "  (minimum at %.1f–%.1f)" % (0.1 * np.argmin(h[5:]) + 0.5, 0.1 * np.argmin(h[5:]) + 0.6))
print("slowest retained flow (time-averaged |v|): %.2f cm/s" % st[vessel].min())
mv = vm[:, tissue].mean(1)
print("mean velocity vector of static tissue (brain voxels outside the vessel set, this frame): (%.2f, %.2f, %.2f), |.| = %.2f cm/s" % (mv[0], mv[1], mv[2], np.linalg.norm(mv)))

print("\n=== Table 4.8  vessel column (peak-systolic frame; |v| time-averaged)")
noise = np.sqrt(8 / np.pi) * a.sigma_v
print("voxels %d | |v| median (time-averaged) %.2f cm/s | |dv| median %.2f, p95 %.2f, max %.2f cm/s | |dv|/|v| median %.1f %%, p95 %.1f %% | at or below noise (time-averaged |v| <= %.2f cm/s): %.1f %%"
      % (vessel.sum(), np.median(st[vessel]), np.median(dabs[vessel]), np.percentile(dabs[vessel], 95), dabs[vessel].max(),
         100 * np.median(rel[vessel]), 100 * np.percentile(rel[vessel], 95), noise, 100 * (st[vessel] <= noise).mean()))

print("\n=== Table 4.9  vessel set binned by distance from the isocentre (peak-systolic frame)")
for lo, hi in [(30, 45), (45, 60), (60, 75), (75, 95), (95, 130)]:
    m = vessel & (R >= lo) & (R < hi)
    print("  %3d–%3d mm: |dv| %.3f cm/s  |dv|/|v| %.2f %%  n %d" % (lo, hi, dabs[m].mean(), 100 * dabs[m].mean() / sm[m].mean(), m.sum()))

print("\n=== Section 4.5.3  patchiness of the signed magnitude change")
dmag = sc - sm; proj = (dv * vm / np.maximum(sm, 1e-9)).sum(0)
print("corr(d|v|, dv . v_hat) on the vessel set: %.3f" % np.corrcoef(dmag[vessel], proj[vessel])[0, 1])
print("direction uncertainty atan(sigma_v / median|v|): %.0f deg  (median |v| of the frame %.1f cm/s)"
      % (np.degrees(np.arctan(a.sigma_v / np.median(sm[vessel]))), np.median(sm[vessel])))
print("voxels with |d|v|| < 0.5 |dv|: %.0f %%" % (100 * np.mean(np.abs(dmag[vessel]) < 0.5 * dabs[vessel])))
acc = np.zeros_like(dmag); cnt = np.zeros_like(dmag)
for ax in range(3):
    for sh in (1, -1):
        acc += np.where(np.roll(vessel, sh, axis=ax), np.roll(dmag, sh, axis=ax), 0); cnt += np.roll(vessel, sh, axis=ax)
ok = vessel & (cnt >= 3)
print("sign of d|v| opposite to the 6-neighbour mean: %.0f %% of vessel voxels" % (100 * np.mean(np.sign(dmag[ok]) != np.sign((acc / np.maximum(cnt, 1))[ok]))))
print("|dv| mean %.2f, std %.2f cm/s" % (dabs[vessel].mean(), dabs[vessel].std()))

def local_rel(f, m):
    fm = np.where(m, f, 0.0); mm = m.astype(float); k = np.ones((3, 3, 3))
    n = ndimage.convolve(mm, k, mode="constant"); s1 = ndimage.convolve(fm, k, mode="constant"); s2 = ndimage.convolve(fm * fm, k, mode="constant")
    okk = m & (n >= 5); lvar = (s2 / n - (s1 / n) ** 2)[okk]
    return float(np.sqrt(np.mean(lvar)) / np.std(f[m]))
for name, f in [("signed d|v|", dmag), ("|dv|", dabs), ("|dv|/|v|", rel)]:
    print(f"local (3x3x3) std / global std, {name:12s}: {local_rel(f, vessel):.2f}")
frac = np.abs(vm) / np.maximum(sm, 1e-9)
for axn, name in [(0, "left-right (x)"), (1, "anterior-posterior (y)"), (2, "superior-inferior (z)")]:
    sel = vessel & (frac[axn] >= 0.8); dd = dmag[sel]
    print(f"{name:24s} n={int(sel.sum()):6d}  mean d|v| {dd.mean():+.3f} cm/s, positive in {100 * np.mean(dd > 0):.0f} %")
