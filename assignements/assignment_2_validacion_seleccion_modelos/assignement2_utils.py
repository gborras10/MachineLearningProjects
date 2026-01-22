import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

def train_model_test_accuracy(
    *,
    model_from_param,   # callable: param_value -> estimator
    param_values,       # iterable of param values
    X_train,
    y_train,
    X_test,
    y_test,
) -> tuple[list, np.ndarray, object, float]:
    """
    Train one model per hyperparameter value and compute test accuracy.

    Returns
    -------
    params : list
        Hyperparameter values (same order as evaluated).
    accuracies : np.ndarray
        Test accuracies for each param.
    best_param : object
        Param with highest accuracy (first occurrence if ties).
    best_acc : float
        Best accuracy.
    """
    params = list(param_values)
    accuracies = np.zeros(len(params), dtype=float)

    for i, p in enumerate(params):
        model = model_from_param(p)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracies[i] = accuracy_score(y_test, y_pred)

    best_idx = int(np.argmax(accuracies))
    return params, accuracies, params[best_idx], float(accuracies[best_idx])


def plot_hyperparameter_sweep(
    *,
    x_values,
    y_values,
    x_label: str,
    y_label: str = "Precisión (accuracy)",
    title: str = "",
    xscale: str = "linear",     # "linear", "log", "symlog"
    xlim: tuple[float, float] | None = None,
    symlog_linthresh: float = 10.0,
    marker: str | None = None,
    linewidth: float = 2.0,
    grid: bool = True,
):
    """Professional-looking 2D plot for a hyperparameter sweep."""
    plt.figure(figsize=(8, 5))
    plt.plot(x_values, y_values, marker=marker, linewidth=linewidth)

    if xscale == "symlog":
        plt.xscale("symlog", linthresh=symlog_linthresh)
    else:
        plt.xscale(xscale)

    if xlim is not None:
        plt.xlim(*xlim)

    plt.xlabel(x_label, fontsize=11)
    plt.ylabel(y_label, fontsize=11)
    if title:
        plt.title(title, fontsize=12)

    if grid:
        plt.grid(True, which="both", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.show()
