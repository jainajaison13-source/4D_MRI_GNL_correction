# Changelog

## v1.0.1 – 2026-11-06

Release used for the submitted thesis.

- New option `--directions-blocked` (dialog checkbox "Directions stored block-wise") for cardiac three-direction series whose pixel frames are stored direction by direction behind headers that kept the interleaved trigger times (post-processed exports such as the DIV11-9 series of the thesis). Default behaviour is unchanged. The layout used is written to the run record as `runtime/dicom_frame_layout` ("interleaved" or "blocked").
- Summary figure (`fig05`): the whole-volume statistics are printed in a text box below the panels instead of inside them.
- Fix: a run without a magnitude series no longer fails in the ROI record (`_roi_record` was undefined).
- Added `Analysis/`: the scripts that make the thesis figures and tables from the toolbox outputs (`run_config.json`, `velocity_field.vtk`), and the DICOM header inventory scripts.
- `run_thesis.sh`: DIV11-9 run uses `--directions-blocked`; the axis-mapping check also accepts the dialog ("gui") as source.

## v1.0.0 – 2026-10-13

First release (DOI 10.5281/zenodo.23348758).
