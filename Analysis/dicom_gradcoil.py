#!/usr/bin/env python3
"""
dicom_gradcoil.py — which scanner and which gradient coil acquired a series.

Reads ONE file per folder (headers only) and prints: manufacturer, model,
software version, field strength, study/acquisition date, and every mention
of a gradient-coil name found in the Siemens private header blocks (CSA
series header / MrPhoenixProtocol, e.g. 'tGradientCoil = "SC72CD"' or an
Impulse/AC-type identifier). It also scans the raw file bytes for the known
coil identifiers, so it works for research re-exports whose CSA block is
missing or renamed.

Usage:  python dicom_gradcoil.py                (uses the folders in DEFAULT_FOLDERS)
        python dicom_gradcoil.py <folder-or-file> [...]
"""
import sys, os, re

DEFAULT_FOLDERS = [
    r"C:\Users\anton\CircleofWillis",
    r"C:\Users\anton\Ideal_Phantom_cardiacCycle\study_3_series_10+11_20180829_DICOM\Velocityseries_11_fs4DMRI_ideal_instat_flow_S2_DIS2D",
    r"C:\Users\anton\PipeFLowMeasurements\FSMRT_3D_0_67ISO_X_FLOW300_0089",
]
COIL_PATTERNS = [rb"tGradientCoil\s*=\s*\"?([A-Za-z0-9_\-]+)", rb"GradCoil[A-Za-z]*\s*=\s*\"?([A-Za-z0-9_\-]+)",
                 rb"(SC72[A-Z]*)", rb"(AS82)", rb"(XR80)", rb"(AC[0-9]{2,3}[A-Z]*)", rb"(Impulse)", rb"(IMPULSE)",
                 rb"(Athena)", rb"(Terra)", rb"(7T Plus|7TPlus|7T\+)"]

try:
    import pydicom
except ImportError:
    print("pydicom is not installed:  pip install pydicom"); sys.exit(1)

def first_file(path):
    if os.path.isfile(path):
        return path
    for root, _, names in os.walk(path):
        for n in sorted(names):
            f = os.path.join(root, n)
            try:
                h = pydicom.dcmread(f, stop_before_pixels=True, force=True)
                if hasattr(h, "Rows"):
                    return f
            except Exception:
                pass
    return None

def report(path):
    print("=" * 78); print(path)
    f = first_file(path)
    if f is None:
        print("  no DICOM image file found"); return
    print("  file:", os.path.basename(f))
    h = pydicom.dcmread(f, stop_before_pixels=True, force=True)
    for tag in ("Manufacturer", "ManufacturerModelName", "SoftwareVersions", "MagneticFieldStrength",
                "StudyDate", "AcquisitionDate", "SeriesDate", "StationName", "DeviceSerialNumber",
                "SequenceName", "ProtocolName", "SeriesDescription"):
        v = getattr(h, tag, None)
        if v is not None:
            print(f"  {tag:24s}: {v}")
    # Siemens private blocks that carry the protocol text
    found = set()
    for de in h.iterall():
        try:
            if de.VR in ("OB", "UN", "LO", "LT", "UT", "SH", "CS") and de.value is not None:
                raw = de.value if isinstance(de.value, (bytes, bytearray)) else str(de.value).encode("latin1", "ignore")
                for pat in COIL_PATTERNS:
                    for m in re.finditer(pat, raw):
                        found.add((str(de.tag), m.group(1).decode("latin1", "ignore")))
        except Exception:
            pass
    # raw-byte scan of the whole file (catches blocks pydicom does not parse)
    with open(f, "rb") as fh:
        raw = fh.read()
    for pat in COIL_PATTERNS:
        for m in re.finditer(pat, raw):
            found.add(("raw", m.group(1).decode("latin1", "ignore")))
    if found:
        print("  gradient-coil / system identifiers found in the file:")
        for tag, name in sorted(found):
            print(f"     {name:12s}  (in {tag})")
    else:
        print("  no gradient-coil identifier found in this file (research export without protocol block?)")

for p in (sys.argv[1:] or DEFAULT_FOLDERS):
    report(p)
