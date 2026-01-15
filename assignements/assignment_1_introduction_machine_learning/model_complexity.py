from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

Config = Any  # e.g., int for trees, tuple[int, ...] for MLP, etc.
Dataset = tuple[np.ndarray, np.ndarray]


@dataclass(slots=True)
class ComplexityStudyResult:
    """
    Result container for a model complexity sweep.

    Attributes
    ----------
    configs:
        Configurations evaluated (aligned with all arrays below).
    complexity:
        Nominal complexity values used as the x-axis (one per config).
    train_error:
        Training classification error, defined as 1 - scorer(y_train, y_pred_train).
        If multiple datasets are provided, this is the aggregated (e.g., mean) train error
        across datasets for each configuration.
    test_error:
        Test classification error, defined as 1 - scorer(y_test, y_pred_test).
        If multiple datasets are provided, this is the aggregated (e.g., mean) test error
        across datasets for each configuration.
    realized_complexity:
        Optional "realized" complexity measured from the fitted model (one per config).
        If not provided, it equals `complexity`. If multiple datasets are provided, this
        is the aggregated realized complexity across datasets.
    best_idx:
        Index of the configuration with minimal (aggregated) `test_error`.
    """

    configs: list[Config]
    complexity: np.ndarray
    train_error: np.ndarray
    test_error: np.ndarray
    realized_complexity: np.ndarray
    best_idx: int


class ModelComplexityStudy:
    """
    Generic model complexity study (train/test error vs complexity), for one or many datasets.

    This class performs a sweep over a set of hyperparameter configurations, fits one model
    per configuration on a train split, evaluates train/test classification error, and stores
    metrics for later inspection/plotting.

    You can initialize it with:
      - a single dataset via (X, y), or
      - multiple datasets via datasets=[(X1,y1), (X2,y2), ...].

    If multiple datasets are provided, `run(...)` trains/evaluates the same sweep on each dataset
    and aggregates errors across datasets (default: mean). The plot methods then visualize these
    aggregated curves.

    Notes
    -----
    - For a single dataset, the train/test split is created once (lazy) and reused across runs.
    - For multiple datasets, each dataset gets its own train/test split inside `run(...)`.
      (This avoids storing multiple splits and keeps the API simple.)
    - Results are sorted by nominal complexity for clean plots and consistent indexing.
    - The default `scorer` is `accuracy_score`; errors are reported as 1 - accuracy.
    """

    def __init__(
        self,
        X: np.ndarray | None = None,
        y: np.ndarray | None = None,
        *,
        datasets: Sequence[Dataset] | None = None,
        make_model: Callable[[Config], Any],
        complexity_of: Callable[[Config], float],
        realized_complexity_of: Callable[[Any, Config], float] | None = None,
        test_size: float = 0.3,
        random_state: int = 0,
        stratify: bool = True,
        scorer: Callable[[np.ndarray, np.ndarray], float] = accuracy_score,
        aggregate: Callable[[np.ndarray], float] = np.mean,
    ) -> None:
        """
        Parameters
        ----------
        X:
            Feature matrix of shape (n_samples, n_features). Used only if `datasets` is None.
        y:
            Target array of shape (n_samples,). Used only if `datasets` is None.
        datasets:
            Optional sequence of datasets, each as (X, y). If provided, it overrides (X, y).
        make_model:
            Callable that builds a fresh (unfitted) model given a configuration.
            The returned object must implement `.fit(X, y)` and `.predict(X)`.
        complexity_of:
            Callable mapping a configuration to a scalar "complexity" value (x-axis).
        realized_complexity_of:
            Optional callable mapping (fitted_model, configuration) to a scalar "realized"
            complexity (e.g., actual tree depth). If None, realized complexity equals nominal.
        test_size:
            Fraction of samples used for the test split (per dataset).
        random_state:
            Random seed used in the train/test split (per dataset).
        stratify:
            Whether to stratify the split by labels `y` (recommended for classification).
        scorer:
            Scoring function with signature `scorer(y_true, y_pred) -> float`.
            Default is `sklearn.metrics.accuracy_score`.
        aggregate:
            Aggregation function applied across datasets for each configuration.
            Must accept a 1D array and return a scalar (e.g., np.mean, np.median, np.max).

        Attributes Set
        --------------
        models_:
            - Single dataset: list of fitted models aligned with sorted configs after `run(...)`.
            - Multiple datasets: list (per dataset) of lists (per config) of fitted models.
        result_:
            `ComplexityStudyResult` produced by the most recent `run(...)`.
        """
        if datasets is None:
            if X is None or y is None:
                raise ValueError("Provide either (X, y) or datasets=[(X1, y1), ...].")
            self.datasets: list[Dataset] = [(X, y)]
        else:
            self.datasets = list(datasets)

        self.make_model = make_model
        self.complexity_of = complexity_of
        self.realized_complexity_of = realized_complexity_of

        self.test_size = test_size
        self.random_state = random_state
        self.stratify = stratify
        self.scorer = scorer
        self.aggregate = aggregate

        # Single-dataset cached split (kept for backwards behavior & speed)
        self._split_done = False
        self.X_train: np.ndarray
        self.X_test: np.ndarray
        self.y_train: np.ndarray
        self.y_test: np.ndarray

        # Models:
        # - if one dataset: list[Any]
        # - if many datasets: list[list[Any]]
        self.models_: Any = []
        self.result_: ComplexityStudyResult | None = None

    def _ensure_split(self) -> None:
        """Create and cache a single train/test split (single-dataset mode only)."""
        if len(self.datasets) != 1:
            return
        if self._split_done:
            return

        X, y = self.datasets[0]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y if self.stratify else None,
        )
        self._split_done = True

    def run(self, configs: Iterable[Config]) -> ComplexityStudyResult:
        """
        Fit and evaluate one model per configuration.

        Parameters
        ----------
        configs:
            Iterable of configurations to evaluate.

        Returns
        -------
        ComplexityStudyResult
            Object containing the evaluated configurations, complexity axis, aggregated
            train/test errors, aggregated realized complexity, and the best index.
        """
        cfgs = list(configs)
        if not cfgs:
            raise ValueError("configs must be non-empty.")

        n_cfg = len(cfgs)
        comp = np.array([float(self.complexity_of(c)) for c in cfgs], dtype=float)

        # Single-dataset mode: keep original behavior (cached split, one model list)
        if len(self.datasets) == 1:
            self._ensure_split()

            train_err = np.empty(n_cfg, dtype=float)
            test_err = np.empty(n_cfg, dtype=float)
            realized = np.empty(n_cfg, dtype=float)

            models: list[Any] = []

            for i, cfg in enumerate(cfgs):
                model = self.make_model(cfg)
                model.fit(self.X_train, self.y_train)
                models.append(model)

                yhat_train = model.predict(self.X_train)
                yhat_test = model.predict(self.X_test)

                score_train = float(self.scorer(self.y_train, yhat_train))
                score_test = float(self.scorer(self.y_test, yhat_test))

                train_err[i] = 1.0 - score_train
                test_err[i] = 1.0 - score_test

                realized[i] = (
                    comp[i]
                    if self.realized_complexity_of is None
                    else float(self.realized_complexity_of(model, cfg))
                )

            # sort by complexity
            order = np.argsort(comp)
            cfgs = [cfgs[i] for i in order]
            comp = comp[order]
            train_err = train_err[order]
            test_err = test_err[order]
            realized = realized[order]
            models = [models[i] for i in order]

            self.models_ = models
            best_idx = int(np.argmin(test_err))

            self.result_ = ComplexityStudyResult(
                configs=cfgs,
                complexity=comp,
                train_error=train_err,
                test_error=test_err,
                realized_complexity=realized,
                best_idx=best_idx,
            )
            return self.result_

        # Multi-dataset mode: compute per-dataset curves and aggregate across datasets
        train_err_ds: list[np.ndarray] = []
        test_err_ds: list[np.ndarray] = []
        realized_ds: list[np.ndarray] = []
        models_all: list[list[Any]] = []

        for (X, y) in self.datasets:
            X_tr, X_te, y_tr, y_te = train_test_split(
                X,
                y,
                test_size=self.test_size,
                random_state=self.random_state,
                stratify=y if self.stratify else None,
            )

            train_err = np.empty(n_cfg, dtype=float)
            test_err = np.empty(n_cfg, dtype=float)
            realized = np.empty(n_cfg, dtype=float)

            models_this_ds: list[Any] = []

            for i, cfg in enumerate(cfgs):
                model = self.make_model(cfg)
                model.fit(X_tr, y_tr)
                models_this_ds.append(model)

                yhat_tr = model.predict(X_tr)
                yhat_te = model.predict(X_te)

                s_tr = float(self.scorer(y_tr, yhat_tr))
                s_te = float(self.scorer(y_te, yhat_te))

                train_err[i] = 1.0 - s_tr
                test_err[i] = 1.0 - s_te

                realized[i] = (
                    comp[i]
                    if self.realized_complexity_of is None
                    else float(self.realized_complexity_of(model, cfg))
                )

            train_err_ds.append(train_err)
            test_err_ds.append(test_err)
            realized_ds.append(realized)
            models_all.append(models_this_ds)

        # aggregate across datasets (per configuration)
        train_mat = np.vstack(train_err_ds)    # shape: (n_datasets, n_cfg)
        test_mat = np.vstack(test_err_ds)
        real_mat = np.vstack(realized_ds)

        train_agg = np.apply_along_axis(self.aggregate, 0, train_mat)
        test_agg = np.apply_along_axis(self.aggregate, 0, test_mat)
        realized_agg = np.apply_along_axis(self.aggregate, 0, real_mat)

        # sort by complexity
        order = np.argsort(comp)
        cfgs = [cfgs[i] for i in order]
        comp = comp[order]
        train_agg = train_agg[order]
        test_agg = test_agg[order]
        realized_agg = realized_agg[order]
        models_all = [[m[i] for i in order] for m in models_all]

        self.models_ = models_all
        best_idx = int(np.argmin(test_agg))

        self.result_ = ComplexityStudyResult(
            configs=cfgs,
            complexity=comp,
            train_error=train_agg,
            test_error=test_agg,
            realized_complexity=realized_agg,
            best_idx=best_idx,
        )
        return self.result_

    @property
    def best_model_(self) -> Any:
        """
        Returns
        -------
        Any
            Fitted model achieving minimal (aggregated) test error in the most recent run.

        Notes
        -----
        - Single dataset: returns the fitted model.
        - Multiple datasets: returns a list of fitted models (one per dataset) for the best config.
        """
        if self.result_ is None:
            raise RuntimeError("Call run(configs) first.")

        if len(self.datasets) == 1:
            return self.models_[self.result_.best_idx]

        # Multi-dataset: return best model per dataset (same best config index)
        return [models_ds[self.result_.best_idx] for models_ds in self.models_]

    @property
    def best_config_(self) -> Config:
        """Configuration achieving minimal (aggregated) test error in the most recent run."""
        if self.result_ is None:
            raise RuntimeError("Call run(configs) first.")
        return self.result_.configs[self.result_.best_idx]

    def plot_error(
        self,
        *,
        title: str,
        xlabel: str,
        ylabel: str = r"$E_{classification}$",
        marker: str = ".",
        odd_xticks: bool = False,
        labels: tuple[str, str] = ("Train error", "Test error"),
    ) -> None:
        """
        Plot (aggregated) train/test classification error vs nominal complexity.

        Parameters
        ----------
        title:
            Plot title.
        xlabel:
            Label for the x-axis (complexity).
        ylabel:
            Label for the y-axis (error).
        marker:
            Matplotlib marker style.
        odd_xticks:
            If True and the x-axis values are (near) integers, show only odd tick labels.
        labels:
            Legend labels for (train, test).

        Raises
        ------
        RuntimeError
            If `run(...)` has not been called yet.
        """
        if self.result_ is None:
            raise RuntimeError("Call run(configs) first.")

        r = self.result_
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(r.complexity, r.train_error, marker=marker, label=labels[0])
        ax.plot(r.complexity, r.test_error, marker=marker, label=labels[1])
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend()

        if odd_xticks and np.all(np.isclose(r.complexity, np.round(r.complexity))):
            ints = r.complexity.astype(int)
            odd = ints[ints % 2 == 1]
            ax.set_xticks(odd)

        plt.show()

    def plot_realized_complexity(
        self,
        *,
        title: str,
        xlabel: str,
        ylabel: str,
        marker: str = "o",
        odd_xticks: bool = False,
    ) -> None:
        """
        Plot (aggregated) realized complexity vs nominal complexity.

        Parameters
        ----------
        title:
            Plot title.
        xlabel:
            Label for the x-axis (nominal complexity).
        ylabel:
            Label for the y-axis (realized complexity).
        marker:
            Matplotlib marker style.
        odd_xticks:
            If True and the x-axis values are (near) integers, show only odd tick labels.

        Raises
        ------
        RuntimeError
            If `run(...)` has not been called yet.
        """
        if self.result_ is None:
            raise RuntimeError("Call run(configs) first.")

        r = self.result_
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(r.complexity, r.realized_complexity, marker=marker)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

        if odd_xticks and np.all(np.isclose(r.complexity, np.round(r.complexity))):
            ints = r.complexity.astype(int)
            odd = ints[ints % 2 == 1]
            ax.set_xticks(odd)

        plt.show()