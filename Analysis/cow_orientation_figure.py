#!/usr/bin/env python3
"""
cow_orientation_figure.py — orientation of the correction in vivo.

Reads the velocity_field.vtk of the in vivo run (exported cardiac phase) and
draws four panels at text width:
  (a) rotation angle theta = angle(v_meas, v_corr) over the vessel tree
      (axial view; each pixel shows the vessel voxel of maximum speed along z)
  (b) the same on the cut-out around the circle of Willis
  (c) dominant flow axis of every vessel voxel on the cut-out (LR / AP / SI),
      with the mean relative correction of each group in the legend
  (d) median rotation angle against distance from the isocentre, for the
      three flow-direction groups and for all vessel voxels
Vessel set as in the thesis (Section 4.5.1): the toolbox's brain mask
(rebuilt with the toolbox's own _build_brain_mask, --toolbox), anatomy
magnitude >= --mag-min and flow coherence >= --coh-min. Nothing is recomputed
from Lambda: every value is read from the export. Axes in mm of the patient frame (LPS), origin
at the magnet isocentre.

Usage:  python cow_orientation_figure.py                               (uses DEFAULT_VTK below)
        python cow_orientation_figure.py velocity_field.vtk out.png [--box x0,x1,y0,y1]
"""
# Default input when the script is started without arguments: the exported
# field of the thesis in vivo run. Edit this line for another machine.
DEFAULT_VTK = r"C:/Users/anton/output_Circle of Willis (In Vivo)/velocity_field.vtk"

import argparse, io, contextlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "DejaVu Sans"   # matplotlib's bundled font: identical output on every machine
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch


def read_vtk(path, names):
    hdr, out = {}, {}
    with open(path, "rb") as f:
        npts = None
        while True:
            line = f.readline()
            if not line:
                break
            s = line.decode("latin1").strip()
            if s.startswith("DIMENSIONS"):
                hdr["dims"] = tuple(int(v) for v in s.split()[1:])
            elif s.startswith("ORIGIN"):
                hdr["origin"] = tuple(float(v) for v in s.split()[1:])
            elif s.startswith("SPACING"):
                hdr["spacing"] = tuple(float(v) for v in s.split()[1:])
            elif s.startswith("POINT_DATA"):
                npts = int(s.split()[1])
            elif s.startswith(("VECTORS", "SCALARS")):
                kind, name = s.split()[0], s.split()[1]
                if kind == "SCALARS":
                    pos = f.tell()
                    if not f.readline().decode("latin1").startswith("LOOKUP_TABLE"):
                        f.seek(pos)
                nc = 3 if kind == "VECTORS" else 1
                nb = npts * nc * 4
                nx, ny, nz = hdr["dims"]
                if name in names:
                    a = np.frombuffer(f.read(nb), dtype=">f4").astype(np.float32)
                    if nc == 3:
                        a = a.reshape(nz, ny, nx, 3).transpose(3, 2, 1, 0)   # (3, nx, ny, nz)
                    else:
                        a = a.reshape(nz, ny, nx).transpose(2, 1, 0)         # (nx, ny, nz)
                    out[name] = np.ascontiguousarray(a)
                else:
                    f.seek(nb, 1)
                f.readline()
    return hdr, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("vtk", nargs="?", default=DEFAULT_VTK,
                    help=f"velocity_field.vtk of the in vivo run (default: {DEFAULT_VTK})")
    ap.add_argument("out", nargs="?", default="cow_orientation.png",
                    help="output image (default: cow_orientation.png)")
    ap.add_argument("--box", default="-45,46.6,-87.6,4.0", help="cut-out x0,x1,y0,y1 in mm (LPS)")
    ap.add_argument("--mag-min", type=float, default=41.1)
    ap.add_argument("--coh-min", type=float, default=0.8)
    ap.add_argument("--toolbox", default=None,
                    help="path to 4D_MRI_GNL_correction*.py (for the brain mask); default: next to this script")
    a = ap.parse_args()

    hdr, d = read_vtk(a.vtk, {"velocity_measured", "velocity_corrected", "speed_measured",
                              "anatomy_magnitude", "flow_coherence"})
    nx, ny, nz = hdr["dims"]; ox, oy, oz = hdr["origin"]; sx, sy, sz = hdr["spacing"]
    v, vc = d["velocity_measured"].astype(np.float64), d["velocity_corrected"].astype(np.float64)
    sm, mag, coh = d["speed_measured"], d["anatomy_magnitude"], d["flow_coherence"]
    keep = (mag >= a.mag_min) & (coh >= a.coh_min)
    tb = Path(a.toolbox) if a.toolbox else None
    if tb is None:
        cands = (sorted(Path(__file__).resolve().parent.glob("4D_MRI_GNL_correction*.py"))
                 or sorted(Path(__file__).resolve().parent.parent.glob("4D_MRI_GNL_correction*.py"))
                 or sorted(Path.cwd().glob("4D_MRI_GNL_correction*.py")))
        tb = cands[0] if cands else None
    if tb is not None and tb.is_file():
        src = tb.read_text(encoding="utf-8")
        i = src.index("def _build_brain_mask"); j = src.index("\ndef ", i + 10)
        ns = {"np": np}; exec(src[i:j], ns)
        with contextlib.redirect_stdout(io.StringIO()):
            bm = ns["_build_brain_mask"](mag)
        keep &= bm
        print(f"brain mask from {tb.name} applied")
    else:
        print("WARNING: toolbox not found, brain mask not applied (vessel set larger than in the thesis)")

    # per-voxel quantities on the vessel set
    nv = np.linalg.norm(v, axis=0); nc_ = np.linalg.norm(vc, axis=0)
    cosang = np.clip((v * vc).sum(0) / np.maximum(nv * nc_, 1e-9), -1, 1)
    theta = np.degrees(np.arccos(cosang))
    dv = vc - v
    rel = 100.0 * np.linalg.norm(dv, axis=0) / np.maximum(nv, 1e-9)
    dom = np.abs(v).argmax(0)                     # 0 = x (LR), 1 = y (AP), 2 = z (SI)
    # "dominant" as in Table 4.10 of the thesis: the component holds >= 80 % of the speed
    strict = np.abs(v).max(0) >= 0.8 * np.maximum(nv, 1e-9)
    xmm = ox + sx * np.arange(nx); ymm = oy + sy * np.arange(ny); zmm = oz + sz * np.arange(nz)
    X, Y, Z = np.meshgrid(xmm, ymm, zmm, indexing="ij")
    r = np.sqrt(X ** 2 + Y ** 2 + Z ** 2)

    # axial pick: the vessel voxel of maximum measured speed along z
    smk = np.where(keep, sm, -np.inf)
    iz = smk.argmax(axis=2); has = keep.any(axis=2)
    ii, jj = np.indices((nx, ny))
    pick = lambda f: np.where(has, f[ii, jj, iz], np.nan)
    th_map, rel_map = pick(theta), pick(rel)
    dom_map = pick(np.where(strict, dom, np.nan).astype(float))   # panel (c): strict groups only
    anat = mag.max(axis=2)
    ext = [xmm[0] - sx / 2, xmm[-1] + sx / 2, ymm[-1] + sy / 2, ymm[0] - sy / 2]
    x0, x1, y0, y1 = (float(t) for t in a.box.split(","))

    # group statistics at matched radius (62-68 mm, as Table 4.10 of the thesis)
    names = ["left–right (x)", "anterior–posterior (y)", "superior–inferior (z)"]
    band = keep & strict & (r >= 62) & (r < 68)
    # mean|dv| / mean|v| per group (the statistic of Table 4.10)
    ndv = np.linalg.norm(dv, axis=0)
    grp_rel = [100.0 * ndv[band & (dom == k)].sum() / nv[band & (dom == k)].sum() for k in range(3)]
    grp_n = [int((band & (dom == k)).sum()) for k in range(3)]
    print("vessel voxels:", int(keep.sum()), " of which at 62-68 mm:", int(band.sum()))
    for k in range(3):
        print(f"  {names[k]:24s} n={grp_n[k]:6d}  mean rel. correction {grp_rel[k]:.2f} %  "
              f"median theta (all radii) {np.median(theta[keep & (dom == k)]):.2f} deg")

    # theta vs radius per group
    edges = np.arange(30, 125, 10)
    centres = 0.5 * (edges[1:] + edges[:-1])
    def prof(mask):
        out = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = mask & (r >= lo) & (r < hi)
            out.append(np.median(theta[m]) if m.sum() >= 50 else np.nan)
        return np.array(out)

    fig, axes = plt.subplots(2, 2, figsize=(11, 11.6))
    thmax = float(np.nanpercentile(th_map, 99))
    # (a) full slab
    ax = axes[0, 0]
    ax.imshow(anat.T, origin="upper", extent=ext, cmap="gray", aspect="equal",
              vmin=np.percentile(anat, 1), vmax=np.percentile(anat, 99.5))
    im = ax.imshow(th_map.T, origin="upper", extent=ext, cmap="viridis", vmin=0, vmax=thmax, aspect="equal")
    ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec="white", lw=1.2, ls="--"))
    ax.set_title("(a) rotation of the velocity vector, whole slab", fontsize=14)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03); cb.set_label("rotation angle θ  [°]", fontsize=12); cb.ax.tick_params(labelsize=11)
    # (b) cut-out
    ax = axes[0, 1]
    ax.imshow(anat.T, origin="upper", extent=ext, cmap="gray", aspect="equal",
              vmin=np.percentile(anat, 1), vmax=np.percentile(anat, 99.5))
    im = ax.imshow(th_map.T, origin="upper", extent=ext, cmap="viridis", vmin=0, vmax=thmax, aspect="equal")
    ax.set_xlim(x0, x1); ax.set_ylim(y1, y0)
    ax.set_title("(b) rotation angle, circle of Willis", fontsize=14)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03); cb.set_label("rotation angle θ  [°]", fontsize=12); cb.ax.tick_params(labelsize=11)
    # (c) dominant flow axis
    ax = axes[1, 0]
    cols = ["#C8651B", "#1F6F8B", "#7B2D8E"]
    cmap3 = ListedColormap(cols); norm3 = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], 3)
    ax.imshow(anat.T, origin="upper", extent=ext, cmap="gray", aspect="equal",
              vmin=np.percentile(anat, 1), vmax=np.percentile(anat, 99.5))
    ax.imshow(dom_map.T, origin="upper", extent=ext, cmap=cmap3, norm=norm3, aspect="equal")
    ax.set_xlim(x0, x1); ax.set_ylim(y1, y0)
    ax.set_title("(c) flow direction (component ≥ 80 % of |v|)", fontsize=14)
    ax.legend(handles=[Patch(color=cols[k], label=f"{names[k]}:  {grp_rel[k]:.2f} %") for k in range(3)],
              loc="lower left", fontsize=10, framealpha=0.9, title="|Δv|/|v| at 62–68 mm from the isocentre", title_fontsize=10)
    # (d) theta vs radius
    ax = axes[1, 1]
    for k in range(3):
        ax.plot(centres, prof(keep & strict & (dom == k)), "o-", color=cols[k], lw=2, ms=5, label=names[k])
    ax.plot(centres, prof(keep), "k--", lw=1.5, label="all vessel voxels")
    ax.set_xlabel("distance from isocentre  r [mm]", fontsize=12)
    ax.set_ylabel("median rotation angle  [°]", fontsize=12)
    ax.set_title("(d) rotation angle against radius, by flow direction", fontsize=14)
    ax.grid(alpha=0.3); ax.legend(fontsize=10); ax.tick_params(labelsize=11)
    for ax in (axes[0, 0], axes[0, 1], axes[1, 0]):
        ax.set_xlabel("x [mm]  (patient left →)", fontsize=12); ax.set_ylabel("y [mm]  (posterior ↓)", fontsize=12)
        ax.tick_params(labelsize=11)
    fig.suptitle("Orientation of the correction in vivo (vessel set, exported cardiac phase)", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(a.out, dpi=200); print("saved", a.out)


if __name__ == "__main__":
    main()
