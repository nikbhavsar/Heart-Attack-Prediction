def train_and_tune_model(
    name, model, param_grid,
    X_train, y_train, X_test, y_test,
    preprocessor, resampler=None,
    scoring_metric='recall', min_recall=0.70
):
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import (
        classification_report, roc_auc_score,
        precision_recall_curve, auc, precision_score,
        recall_score, f1_score
    )
    from imblearn.pipeline import Pipeline as ImbPipeline
    import matplotlib.pyplot as plt
    import numpy as np

    print(f"\n================= {name} =================")

    steps = [('preprocess', preprocessor)]
    if resampler:
        steps.append(('resample', resampler))
    steps.append(('clf', model))
    pipe = ImbPipeline(steps)

    grid = GridSearchCV(pipe, param_grid, scoring=scoring_metric, cv=3, n_jobs=1)
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    y_proba = best_model.predict_proba(X_test)[:, 1]  # Prob for class 1 (heart attack)

    precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba, pos_label=1)
    pr_auc = auc(recalls, precisions)

    # Threshold selection: recall ≥ min_recall
    valid_idxs = np.where(recalls >= min_recall)[0]

    if len(valid_idxs) == 0:
        print(f"No threshold found where recall ≥ {min_recall:.2f}.")
        best_thresh = 0.5
    else:
        best_thresh_idx = valid_idxs[np.argmax(precisions[valid_idxs])]
        best_thresh = thresholds[best_thresh_idx]
        print(f"Threshold achieving recall ≥ {min_recall:.2f}: {best_thresh:.2f}")

    y_pred = (y_proba >= best_thresh).astype(int)

    precision_1 = precision_score(y_test, y_pred, pos_label=1)
    recall_1 = recall_score(y_test, y_pred, pos_label=1)
    f1_1 = f1_score(y_test, y_pred, pos_label=1)
    roc = roc_auc_score(y_test, y_proba)

    print("Best Params:", grid.best_params_)
    print(f"Threshold Used: {best_thresh:.2f}")
    print(f"Recall (Heart Attack): {recall_1:.2f}")
    print(f"Precision (Heart Attack): {precision_1:.2f}")
    print(f"F1 Score (Heart Attack): {f1_1:.2f}")
    print(f"ROC AUC: {roc:.2f}")
    print(f"PR AUC (Heart Attack): {pr_auc:.2f}")
    print("Full Report:\n", classification_report(y_test, y_pred, target_names=["No Heart Attack", "Heart Attack"]))

    plt.plot(recalls, precisions, label=f'{name} (PR AUC = {pr_auc:.2f})')
    plt.xlabel("Recall (Heart Attack)")
    plt.ylabel("Precision (Heart Attack)")
    plt.title("Precision-Recall Curve (Heart Attack Class)")
    plt.legend()

    return {
        'name': name,
        'model': best_model,
        'params': grid.best_params_,
        'threshold': best_thresh,
        'precision': precision_1,
        'recall': recall_1,
        'f1': f1_1,
        'roc_auc': roc,
        'pr_auc': pr_auc
    }
