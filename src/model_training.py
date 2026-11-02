"""Model determination, training and tuning (Tasks 3–5)."""
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold

# FIX: import the sibling module from inside the src package
from src.preprocessing import build_preprocessor


def get_models(scale_numeric_for_lr: bool = True):
    """Return dict of candidate models (Pipeline(preprocessor -> estimator)).

    Model reasoning (Task 3):
      - Logistic Regression: interpretable baseline; coefficients show
        direction of effect; works well with Credit_History.
      - Random Forest: bagged trees; handles non-linearity & interactions;
        robust to outliers; gives feature importances.
      - Gradient Boosting: sequential trees correcting prior errors;
        often highest accuracy on tabular data but prone to overfit
        without tuning.
    """
    models = {
        "LogisticRegression": Pipeline([
            ("pre", build_preprocessor(scale_numeric=scale_numeric_for_lr)),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced",
                                       random_state=42)),
        ]),
        "RandomForest": Pipeline([
            ("pre", build_preprocessor(scale_numeric=False)),
            ("clf", RandomForestClassifier(random_state=42,
                                           class_weight="balanced")),
        ]),
        "GradientBoosting": Pipeline([
            ("pre", build_preprocessor(scale_numeric=False)),
            ("clf", GradientBoostingClassifier(random_state=42)),
        ]),
    }
    return models


PARAM_GRIDS = {
    "LogisticRegression": {
        "clf__C": [0.01, 0.1, 1, 10],
        "clf__solver": ["liblinear", "lbfgs"],
    },
    "RandomForest": {
        "clf__n_estimators": [100, 200, 400],
        "clf__max_depth": [None, 5, 10],
        "clf__min_samples_leaf": [1, 3],
    },
    "GradientBoosting": {
        "clf__n_estimators": [100, 200],
        "clf__learning_rate": [0.05, 0.1, 0.2],
        "clf__max_depth": [2, 3],
    },
}


def tune_model(name, model, X_train, y_train, cv_folds: int = 5):
    """Grid search with 5-fold CV on the TRAINING set only (no leakage)."""
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    gs = GridSearchCV(
        model, PARAM_GRIDS[name], cv=cv,
        scoring={"acc": "accuracy", "f1": "f1", "roc": "roc_auc"},
        refit="f1", n_jobs=-1, verbose=0,
    )
    gs.fit(X_train, y_train)
    return gs