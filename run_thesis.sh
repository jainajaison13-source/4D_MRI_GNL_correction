#!/usr/bin/env bash
# run_thesis.sh - the exact toolbox commands used for the runs reported in the thesis
#   "Open-Source Preprocessing for Gradient Nonlinearity Correction in 4D Flow MRI"
#   (Jaina Jaison, Otto von Guericke University Magdeburg, 2026).
#
# Usage (Git Bash or any bash, from the folder that holds this script and the toolbox):
#     bash run_thesis.sh pipe0081 | pipe0080 | pipe0089 | pipe0088 | phantom11 | div11 | cow | impulse
#
# Before the first run, set DATA_ROOT and the coefficient-file paths below to your own locations.
# The coefficient (.grad) files are the manufacturer's property and are not part of this repository.
#
# Never start the toolbox without arguments for the phantom and in vivo datasets: the file
# dialog cannot set the velocity-axis mapping (--direction-permute / --direction-signs), and
# a run without it comes out wrong (see Section 3.6.1 of the thesis).

set -u
cd "$(dirname "$0")"

TOOLBOX="4D_MRI_GNL_correction.py"

# ---- edit these paths ----------------------------------------------------------
DATA_ROOT="C:/Users/<you>"                         # folder that holds the datasets
GRAD_3T="$DATA_ROOT/coeff_skyra.grad"             # 3 T MAGNETOM Skyra coil file
GRAD_7T="$DATA_ROOT/coeff_SC72.grad"              # 7 T SC72CD coil file
GRAD_IMPULSE="$DATA_ROOT/coeff_IMPULSE.grad"      # Impulse head gradient coil file
PIPE_DIR="$DATA_ROOT/PipeFlowMeasurements"        # contains FSMRT_3D_1ISO_X_FLOW300_00xx
PH11_DIR="$DATA_ROOT/Ideal_Phantom_cardiacCycle/study_3_series_10+11_20180829_DICOM/Velocityseries_11_fs4DMRI_ideal_instat_flow_S2_DIS2D"
DIV11_DIR="$DATA_ROOT/Ideal_Phantom_cardiacCycle/study_3_series_DIV11-9_20180829_DICOM"
COW_DIR="$DATA_ROOT/CircleofWillis"
OUT_ROOT="$DATA_ROOT/output"                      # one output folder per run is created below
# ----------------------------------------------------------------------------------

MAG=""
case "${1:-}" in
  pipe0081|pipe0080|pipe0089|pipe0088)
    SERIES="${1#pipe}"
    ARGS=(--grad "$GRAD_3T" --dicom "$PIPE_DIR/FSMRT_3D_1ISO_X_FLOW300_$SERIES"
          --preset pipe_flow --venc 0.07 --sim-mode 1 --out "$OUT_ROOT/pipe_$SERIES") ;;
  phantom11)
    ARGS=(--grad "$GRAD_7T" --dicom "$PH11_DIR" --preset ideal_phantom --venc 1.5
          --direction-permute yxz "--direction-signs=1,1,-1" --out "$OUT_ROOT/phantom_series11") ;;
  div11)
    ARGS=(--grad "$GRAD_7T" --dicom "$DIV11_DIR" --preset ideal_phantom --venc 1.5
          --direction-permute yxz "--direction-signs=1,1,-1" --out "$OUT_ROOT/phantom_DIV11-9") ;;
  cow)
    ARGS=(--grad "$GRAD_7T" --dicom "$COW_DIR" --preset in_vivo_head --venc 0.9
          --direction-permute yxz "--direction-signs=-1,-1,1" --out "$OUT_ROOT/CoW_in_vivo") ;;
  impulse)
    # coil characterisation and synthetic verification only: no DICOM data exist for this coil
    ARGS=(--grad "$GRAD_IMPULSE" --preset ideal_phantom --out "$OUT_ROOT/impulse_simulation") ;;
  *)
    echo "Usage: bash run_thesis.sh pipe0081 | pipe0080 | pipe0089 | pipe0088 | phantom11 | div11 | cow | impulse"; exit 1 ;;
esac

LOG="log_$1.txt"
echo "Running: python -u $TOOLBOX ${ARGS[*]}"
echo "Full output is also saved to $LOG"
echo
python -u "$TOOLBOX" "${ARGS[@]}" 2>&1 | tee "$LOG"

# ---- for the DICOM runs: check that the axis mapping really reached the run ----------------
case "$1" in phantom11|div11|cow)
OUT="${ARGS[${#ARGS[@]}-1]}"
echo
python - "$OUT/run_config.json" <<'PY'
import json, sys
try:
    d = json.load(open(sys.argv[1], encoding="utf-8"))
except Exception as e:
    print(f"CHECK FAILED: cannot read {sys.argv[1]} ({e})"); sys.exit(1)
dc = d.get("dicom_correction")
if not dc:
    print("CHECK FAILED: no DICOM results in run_config.json - the DICOM stage did not finish.")
    print("              Look for 'STAGE 5 FAILED' in the log file.")
    sys.exit(1)
af = dc.get("axis_flags", {})
ok = af.get("permute_applied") and af.get("signs_applied") and af.get("source") == "cli"
print("AXIS MAPPING CHECK:", "OK" if ok else "FAILED", "-", af)
sys.exit(0 if ok else 1)
PY
;; esac
