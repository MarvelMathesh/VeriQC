"""VeriQC Ablations Suite.

Formal, reproducible ablations establishing the empirical justification for each
architectural choice in VeriQC:

A1: Cascade Strategy (YOLO vs Anomaly vs Cascade)
A2: Conformal Triage Dial vs Fixed Percentile
A3: Image QC Gate Robustness under Lighting Drift
A4: Advisory Verification (Slot-filling + Verifier vs Raw LLM vs Template)
A5: Anomaly Backbone & Scorer Pareto Frontier (EfficientNet-B0 vs MobileNet vs PaDiM)
A6: Detector Architecture Bake-off (YOLO26n vs YOLO11n vs Faster R-CNN vs SSD)
"""

