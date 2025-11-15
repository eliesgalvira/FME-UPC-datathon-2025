"""Model training modules for teacher and student models.

This package provides:
- histos_sampling.py: HistOS-like sampling for whale modeling
- teacher_classifier.py: CatBoost buyer classifier training
- teacher_regressor.py: LightGBM revenue regressor training
- student_trainer.py: Distillation training for student models
"""

from __future__ import annotations

__all__ = ["histos_sampling", "teacher_classifier", "teacher_regressor", "student_trainer"]



