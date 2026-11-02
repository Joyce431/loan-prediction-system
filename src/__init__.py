"""Loan prediction source package.

Modules:
    data_cleaning  — Task 1: loading, inspection, descriptive stats, outliers, cleaning
    preprocessing  — Task 2: feature extraction/engineering + sklearn ColumnTransformer
    model_training — Tasks 3–5: model selection, grid-search tuning
    evaluate       — Task 4/8a: metrics, confusion matrices, ROC, feature importances

Typical use:
    from src.data_cleaning import load_data, clean, descriptive_stats, outlier_report
    from src.preprocessing import engineer_features, build_preprocessor
    from src.model_training import get_models, tune_model
    from src.evaluate import metrics_dict, save_confusion_matrix, save_roc_curves
"""

__version__ = "1.0.0"
__all__ = [
    "data_cleaning",
    "preprocessing",
    "model_training",
    "evaluate",
]