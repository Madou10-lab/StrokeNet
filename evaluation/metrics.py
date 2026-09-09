"""Run-level metrics and the original ensemble search tie-breaking rule."""
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score


def compute_metrics(score_matrix, labels):
    labels = np.asarray(labels, dtype=np.int64)
    predictions = np.argmax(score_matrix, axis=1)
    top5_indices = np.argsort(score_matrix, axis=1)[:, -5:]
    class_ids = list(range(score_matrix.shape[1]))
    confusion = confusion_matrix(labels, predictions, labels=class_ids)
    support = confusion.sum(axis=1)
    each_acc = np.divide(
        np.diag(confusion),
        support,
        out=np.zeros_like(support, dtype=np.float64),
        where=support != 0,
    )
    return {
        "top1": float(np.mean(predictions == labels)),
        "macro_acc": float(each_acc.mean()),
        "macro_f1": float(
            f1_score(
                labels, predictions, labels=class_ids, average="macro", zero_division=0
            )
        ),
        "top5": float(
            np.mean([label in row for label, row in zip(labels, top5_indices)])
        ),
        "confusion": confusion,
        "each_acc": each_acc,
    }


def is_better(trial, current):
    return (
        trial["top1"] > current["top1"]
        or (
            trial["top1"] == current["top1"] and trial["macro_f1"] > current["macro_f1"]
        )
        or (
            trial["top1"] == current["top1"]
            and trial["macro_f1"] == current["macro_f1"]
            and (trial["top5"] > current["top5"])
        )
    )


def paper_metrics(scores, labels):
    metrics = compute_metrics(scores, labels)
    metrics["top2"] = float(
        np.mean(
            np.any(
                np.argsort(scores, axis=1)[:, -2:] == np.asarray(labels)[:, None],
                axis=1,
            )
        )
    )
    return metrics
