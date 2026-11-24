Archived design-role bake-offs.

These records were produced by the earlier cascade/object-level architecture on
KolektorSDD design and defect-validation roles, before the final profile-isolated
segmentation/PatchCore design. They are retained as model-selection provenance.
The current operating points are reported in the main evidence tables. Each JSON
carries its own provenance, commit and role metadata.

Files:
  detector_bakeoff.json       YOLO26n 640/320 and YOLO26s 640
  detector_training/          per-arm training records
  backbone_pareto.json        EfficientNet-B0, MobileNetV3, ResNet-18, WideResNet-50-2
  anomaly_sweep.json          input size, resize mode and bank-size sweep
  gate_sweep.json             repeated input/resize/bank sweep
  cascade_bakeoff.json        production vs gated-cascade arms
  rule_bakeoff.json           anomaly-only vs composed rule arms
  conformal_dial.json         object-level vs frame-level calibration feasibility
