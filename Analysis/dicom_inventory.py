#!/usr/bin/env python3
"""
dicom_inventory.py — what is in a 4D-flow DICOM folder, from the headers only.

Prints, per folder: number of files, series, ImageType, SequenceName and its
velocity-direction suffix, the distinct TriggerTimes, InstanceNumber range,
and a table "suffix x TriggerTime -> file count" that shows how the three
velocity directions are encoded. No pixel data is read.

Usage:
    python dicom_inventory.py                      -> uses the folders listed in DEFAULT_FOLDERS
    python dicom_inventory.py <folder> [<folder>]  -> uses the given folders

The printout is also written to dicom_inventory.txt next to this script.
"""
import sys, os, re, collections

DEFAULT_FOLDERS = [
    r"C:\Users\anton\Ideal_Phantom_cardiacCycle\study_3_series_10+11_20180829_DICOM\Velocityseries_11_fs4DMRI_ideal_instat_flow_S2_DIS2D",
    r"C:\Users\anton\Ideal_Phantom_cardiacCycle\study_3_series_DIV11-9_20180829_DICOM\study_3_series_DIV11-9_20180829_DICOM",
]

try:
    import pydicom
except ImportError:
    print("pydicom is not installed:  pip install pydicom"); sys.exit(1)

_lines = []
def out(s=""):
    print(s); _lines.append(s)

def inventory(folder):
    out("=" * 78)
    out(folder)
    if not os.path.isdir(folder):
        out("  FOLDER NOT FOUND"); return
    files = []
    for root, _, names in os.walk(folder):
        for n in names:
            files.append(os.path.join(root, n))
    files.sort()
    out(f"  {len(files)} files")
    series, itype, seq, suffix, trig, inst, desc, priv = (collections.Counter() for _ in range(8))
    suf_trig = collections.Counter()
    n_ok = 0
    for f in files:
        try:
            h = pydicom.dcmread(f, stop_before_pixels=True, force=True)
        except Exception:
            continue
        if not hasattr(h, "Rows"):
            continue
        n_ok += 1
        series[str(getattr(h, "SeriesInstanceUID", "?"))[-10:]] += 1
        itype["\\".join(str(x) for x in getattr(h, "ImageType", ["?"]))] += 1
        s = str(getattr(h, "SequenceName", "?"))
        seq[s] += 1
        m = re.search(r"v\d+([a-z]{1,2})$", s.lower())
        sf = m.group(1) if m else "-"
        suffix[sf] += 1
        t = float(getattr(h, "TriggerTime", -1))
        trig[t] += 1
        suf_trig[(sf, t)] += 1
        inst[int(getattr(h, "InstanceNumber", -1))] += 1
        desc[str(getattr(h, "SeriesDescription", "?"))] += 1
        p = h.get((0x0051, 0x1016), None)
        priv[str(p.value) if p is not None else "-"] += 1
    out(f"  readable DICOM image files: {n_ok}")
    if n_ok == 0:
        return
    out(f"  SeriesInstanceUID (last 10 chars) : {dict(series)}")
    out(f"  SeriesDescription                 : {dict(desc)}")
    out(f"  ImageType                         : {dict(itype)}")
    out(f"  private ImageTypeText (0051,1016) : {dict(priv)}")
    out(f"  SequenceName                      : {dict(seq)}")
    out(f"  direction suffix of SequenceName  : {dict(suffix)}")
    tt = sorted(trig)
    out(f"  distinct TriggerTimes             : {len(tt)}  -> {tt[:15]}{' ...' if len(tt) > 15 else ''}")
    out(f"  InstanceNumber range              : {min(inst)} .. {max(inst)}")
    out("  files per (direction suffix, TriggerTime):")
    for (sf, t) in sorted(suf_trig):
        out(f"     suffix {sf:>2}   TriggerTime {t:8.2f}   {suf_trig[(sf, t)]:4d} files")

folders = sys.argv[1:] or DEFAULT_FOLDERS
for folder in folders:
    inventory(folder)

_txt = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dicom_inventory.txt")
with open(_txt, "w", encoding="utf-8") as fh:
    fh.write("\n".join(_lines) + "\n")
print(f"\n(written to {_txt})")
