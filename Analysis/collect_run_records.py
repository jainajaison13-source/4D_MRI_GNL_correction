#!/usr/bin/env python3
"""
collect_run_records.py - gather the small result files of every toolbox run
into one folder tree that can be dragged into the GitHub repository.

For each run it copies ONLY: run_config.json, the five figures fig01..fig05
(*.png) and the log file(s). The large outputs (dicom_corrected.npz,
velocity_field.vtk) and any DICOM data are deliberately NOT copied.

Edit the RUNS dictionary below so that each entry points to the output
folder of that run (the folder that contains run_config.json), then run:

    python collect_run_records.py

Result:  C:/Users/anton/github_upload/run_records/<run name>/...
Then on GitHub: Add file -> Upload files -> drag the whole run_records folder.
"""
import shutil
from pathlib import Path

TARGET = Path(r"C:/Users/anton/github_upload/run_records")

# run name in the repository  ->  toolbox output folder of that run
RUNS = {
    "pipe_0081":           r"C:/Users/anton/Output_Pipe Flow Measurements/FSMRT_3D_1ISO_X_FLOW300_0081",
    "pipe_0080":           r"C:/Users/anton/Output_Pipe Flow Measurements/FSMRT_3D_1ISO_X_FLOW300_0080",
    "pipe_0089":           r"C:/Users/anton/Output_Pipe Flow Measurements/FSMRT_3D_1ISO_X_FLOW300_0089",
    "pipe_0088":           r"C:/Users/anton/Output_Pipe Flow Measurements/FSMRT_3D_1ISO_X_FLOW300_0088",
    "phantom_series11":    r"C:/Users/anton/output_Ideal Phantom –Series11-9_Cardiac Cycle/series11",
    "phantom_DIV11-9":     r"C:/Users/anton/output_Ideal Phantom –Series11-9_Cardiac Cycle/DIV11-9",
    "CoW_in_vivo":         r"C:/Users/anton/output_october/CircleofWillis_in_vivo_head",
    "impulse_simulation":  r"C:/Users/anton/output_october",
}

KEEP = ("run_config.json",)          # always
KEEP_GLOBS = ("fig0*.png", "log*.txt")   # small, useful
SKIP_GLOBS = ("*.npz", "velocity_field*.vtk", "*.dcm", "*.IMA")


def main():
    TARGET.mkdir(parents=True, exist_ok=True)
    for name, folder in RUNS.items():
        src = Path(folder)
        dst = TARGET / name
        if not (src / "run_config.json").is_file():
            print(f"[{name}] SKIPPED - no run_config.json in {src}  (edit the path in RUNS)")
            continue
        dst.mkdir(parents=True, exist_ok=True)
        copied = []
        for f in KEEP:
            shutil.copy2(src / f, dst / f); copied.append(f)
        for pattern in KEEP_GLOBS:
            for f in sorted(src.glob(pattern)):
                shutil.copy2(f, dst / f.name); copied.append(f.name)
        big = [f.name for p in SKIP_GLOBS for f in src.glob(p)]
        print(f"[{name}] copied {len(copied)} files: {', '.join(copied)}")
        if big:
            print(f"           left out (large / data): {', '.join(big)}")
    print(f"\nDone. Drag the folder  {TARGET}  onto GitHub: Add file -> Upload files.")


if __name__ == "__main__":
    main()
