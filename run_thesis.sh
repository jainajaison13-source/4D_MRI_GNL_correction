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
# The thesis runs were made through the start dialog (python 4D_MRI_GNL_correction.py without
# arguments), whose fields correspond one-to-one to the options below (thesis Table 3.4 and
# Fig. 3.2); the run records store the dialog entries. This script gives the same runs as
# command lines. The axis-mapping check at the end accepts both forms (source "cli"/"gui").

set -u
cd "$(dirname "$0")"

TOOLBOX="4D_MRI_GNL_correction.py"

# ---- edit these paths ----------------------------------------------------------
# Folder names are those of the run records of the thesis (runtime block of run_config.json).
DATA_ROOT="C:/Users/<you>"                        # folder that holds the datasets
GRAD_3T="$DATA_ROOT/coeff_skyra.grad.txt"             # 3 T MAGNETOM Skyra coil file
GRAD_7T="$DATA_ROOT/coeff_SC72.grad.txt"              # 7 T SC72CD coil file
GRAD_IMPULSE="$DATA_ROOT/coeff_IMPULSE.grad.txt"      # Impulse head gradient coil file
PIPE_DIR="$DATA_ROOT/PipeFLowMeasurements"
PH_DIR="$DATA_ROOT/Ideal_Phantom_cardiacCycle"
PH11_DIR="$PH_DIR/study_3_series_10+11_20180829_DICOM/Velocityseries_11_fs4DMRI_ideal_instat_flow_S2_DIS2D"
PH10_MAG="$PH_DIR/study_3_series_10+11_20180829_DICOM/magnitudeseries_10_fs4DMRI_ideal_instat_flow_S3_DIS2D"
DIV11_DIR="$PH_DIR/study_3_series_DIV11-9_20180829_DICOM/study_3_series_DIV11-9_20180829_DICOM"
COW_DIR="$DATA_ROOT/CircleofWillis"
COW_MAG="$COW_DIR/ww25_Magnitude_venc0_9.dcm"      # the single multiframe magnitude file of the export
OUT_ROOT="$DATA_ROOT/output"                      # one output folder per run is created below
# Pipe-flow series (velocity folder : magnitude folder of the same acquisition, the next series number)
PIPE_0081="FSMRT_3D_1ISO_X_FLOW300_0081";      PIPE_0081_MAG="FSMRT_3D_1ISO_X_FLOW300_0083"
PIPE_0080="FSMRT_3D_1ISO_X_FLOW300_ND_0080";   PIPE_0080_MAG="FSMRT_3D_1ISO_X_FLOW300_ND_0082"
PIPE_0089="FSMRT_3D_0_67ISO_X_FLOW300_0089";   PIPE_0089_MAG="FSMRT_3D_0_67ISO_X_FLOW300_0091"
PIPE_0088="FSMRT_3D_0_67ISO_X_FLOW300_ND_0088"; PIPE_0088_MAG="FSMRT_3D_0_67ISO_X_FLOW300_ND_0090"
# Analysis box of the aneurysm-phantom runs (voxel indices x0,x1,y0,y1,z0,z1; statistics only)
PH_BOX="40,104,0,223,0,87"
# ----------------------------------------------------------------------------------

case "${1:-}" in
  pipe0081|pipe0080|pipe0089|pipe0088)
    S="${1#pipe}"; V="PIPE_$S"; M="PIPE_${S}_MAG"
    ARGS=(--grad "$GRAD_3T" --dicom "$PIPE_DIR/${!V}" --magnitude "$PIPE_DIR/${!M}"
          --preset pipe_flow --venc 0.07 --sim-mode 1 --out "$OUT_ROOT/pipe_$S") ;;
  phantom11)
    ARGS=(--grad "$GRAD_7T" --dicom "$PH11_DIR" --magnitude "$PH10_MAG" --preset ideal_phantom
          --venc 1.5 --sim-mode 2 --direction-permute yxz "--direction-signs=1,1,-1"
          --roi-box "$PH_BOX" --out "$OUT_ROOT/phantom_series11") ;;
  div11)
    # DIV11-9 is a MeVisLab export whose pixel frames are stored block-wise (all cardiac
    # phases of direction 1, then 2, then 3) behind the interleaved headers of series 11:
    # --directions-blocked assigns the frames accordingly (thesis Sections 3.5 and 3.6.1).
    ARGS=(--grad "$GRAD_7T" --dicom "$DIV11_DIR" --magnitude "$PH10_MAG" --preset ideal_phantom
          --venc 1.5 --sim-mode 2 --direction-permute yxz "--direction-signs=1,1,-1"
          --roi-box "$PH_BOX" --directions-blocked --out "$OUT_ROOT/phantom_DIV11-9") ;;
  cow)
    ARGS=(--grad "$GRAD_7T" --dicom "$COW_DIR" --magnitude "$COW_MAG" --preset in_vivo_head
          --venc 0.9 --sim-mode 2 --direction-permute yxz "--direction-signs=-1,-1,1"
          --out "$OUT_ROOT/CoW_in_vivo") ;;
  impulse)
    # coil characterisation and synthetic verification only: no DICOM data exist for this coil
    ARGS=(--grad "$GRAD_IMPULSE" --preset ideal_phantom --sim-mode 2 --out "$OUT_ROOT/impulse_simulation") ;;
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
ok = af.get("permute_applied") and af.get("signs_applied") and af.get("source") in ("cli", "gui")
print("AXIS MAPPING CHECK:", "OK" if ok else "FAILED", "-", af)
sys.exit(0 if ok else 1)
PY
;; esac
