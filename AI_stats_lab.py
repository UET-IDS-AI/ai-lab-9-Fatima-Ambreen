import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


# ============================================================
# Question 1: Confusion Matrix, Metrics, and Threshold Effects
# ============================================================

def confusion_matrix_counts(y_true, y_pred):

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    TP = np.sum((y_true == 1) & (y_pred == 1))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    TN = np.sum((y_true == 0) & (y_pred == 0))

    return (TP, FP, FN, TN)


def classification_metrics(y_true, y_pred):

    TP, FP, FN, TN = confusion_matrix_counts(y_true, y_pred)

    # Recall
    if (TP + FN) == 0:
        recall = 0.0
    else:
        recall = TP / (TP + FN)

    # Fallout
    if (FP + TN) == 0:
        fallout = 0.0
    else:
        fallout = FP / (FP + TN)

    # Precision
    if (TP + FP) == 0:
        precision = 0.0
    else:
        precision = TP / (TP + FP)

    # Accuracy
    total = TP + FP + FN + TN

    if total == 0:
        accuracy = 0.0
    else:
        accuracy = (TP + TN) / total

    return {
        "recall": recall,
        "fallout": fallout,
        "precision": precision,
        "accuracy": accuracy
    }


def apply_threshold(scores, threshold):

    scores = np.array(scores)

    return np.where(scores >= threshold, 1, 0)


def threshold_metrics_analysis(y_true, scores, thresholds):

    results = []

    for threshold in thresholds:

        preds = apply_threshold(scores, threshold)

        metrics = classification_metrics(y_true, preds)

        result = {
            "threshold": threshold,
            "recall": metrics["recall"],
            "fallout": metrics["fallout"],
            "precision": metrics["precision"],
            "accuracy": metrics["accuracy"]
        }

        results.append(result)

    return results


# ============================================================
# Question 2: Train Two Classifiers and Evaluate Them
# ============================================================

def train_two_classifiers(X_train, y_train):

    # Logistic Regression
    lr_model = LogisticRegression(max_iter=1000)

    # Decision Tree
    dt_model = DecisionTreeClassifier(random_state=0)

    # Train both models
    lr_model.fit(X_train, y_train)
    dt_model.fit(X_train, y_train)

    return {
        "logistic_regression": lr_model,
        "decision_tree": dt_model
    }


def evaluate_classifier(model, X_test, y_test, threshold=0.5):

    # Probabilities for positive class
    scores = model.predict_proba(X_test)[:, 1]

    # Convert to predictions
    y_pred = apply_threshold(scores, threshold)

    # Confusion matrix
    TP, FP, FN, TN = confusion_matrix_counts(y_test, y_pred)

    # Metrics
    metrics = classification_metrics(y_test, y_pred)

    return {
        "TP": TP,
        "FP": FP,
        "FN": FN,
        "TN": TN,
        "recall": metrics["recall"],
        "fallout": metrics["fallout"],
        "precision": metrics["precision"],
        "accuracy": metrics["accuracy"]
    }


def compare_classifiers(X_train, y_train, X_test, y_test, threshold=0.5):

    models = train_two_classifiers(X_train, y_train)

    lr_results = evaluate_classifier(
        models["logistic_regression"],
        X_test,
        y_test,
        threshold
    )

    dt_results = evaluate_classifier(
        models["decision_tree"],
        X_test,
        y_test,
        threshold
    )

    return {
        "logistic_regression": lr_results,
        "decision_tree": dt_results
    }


if __name__ == "__main__":
    print("Implement all required functions.")
