from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

Config = Any
Dataset = tuple[np.ndarray, np.ndarray]


@dataclass(slots=True)
class ComplexityStudyResult:
    configs: list[Config]
    complexity: np.ndarray
    train_error: np.ndarray
    test_error: np.ndarray
    realized_complexity: np.ndarray
    best_idx: int


class ModelComplexityStudy:
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
        scale: bool = True,
    ) -> None:
        if datasets is None:
            if X is None or y is None:
                raise ValueError("Provide either (X, y) or datasets=[...].")
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
        self.scale = scale

        self._split_cached = False
        self.X_train: np.ndarray
        self.X_test: np.ndarray
        self.y_train: np.ndarray
        self.y_test: np.ndarray
        self.scaler_: StandardScaler | None = None

        self.models_: Any = []
        self.result_: ComplexityStudyResult | None = None

    def _scale_split(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, StandardScaler | None]:
        if not self.scale:
            return X_train, X_test, None
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        return X_train_s, X_test_s, scaler

    def _ensure_cached_split(self) -> None:
        if len(self.datasets) != 1 or self._split_cached:
            return

        X_all, y_all = self.datasets[0]
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_all,
            y_all,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y_all if self.stratify else None,
        )
        X_tr, X_te, scaler = self._scale_split(X_tr, X_te)

        self.X_train, self.X_test = X_tr, X_te
        self.y_train, self.y_test = y_tr, y_te
        self.scaler_ = scaler
        self._split_cached = True

    def run(self, configs: Iterable[Config]) -> ComplexityStudyResult:
        cfg_list = list(configs)
        if not cfg_list:
            raise ValueError("configs must be non-empty.")

        comp = np.array(
            [float(self.complexity_of(cfg)) for cfg in cfg_list],
            dtype=float,
        )

        if len(self.datasets) == 1:
            self._ensure_cached_split()
            return self._run_single_dataset(cfg_list, comp)

        return self._run_multi_dataset(cfg_list, comp)

    def _run_single_dataset(
        self,
        cfg_list: list[Config],
        comp: np.ndarray,
    ) -> ComplexityStudyResult:
        n_cfg = len(cfg_list)
        train_err = np.empty(n_cfg, dtype=float)
        test_err = np.empty(n_cfg, dtype=float)
        real_comp = np.empty(n_cfg, dtype=float)
        models: list[Any] = []

        for i, cfg in enumerate(cfg_list):
            model = self.make_model(cfg)
            model.fit(self.X_train, self.y_train)
            models.append(model)

            yhat_tr = model.predict(self.X_train)
            yhat_te = model.predict(self.X_test)

            score_tr = float(self.scorer(self.y_train, yhat_tr))
            score_te = float(self.scorer(self.y_test, yhat_te))

            train_err[i] = 1.0 - score_tr
            test_err[i] = 1.0 - score_te

            if self.realized_complexity_of is None:
                real_comp[i] = comp[i]
            else:
                real_comp[i] = float(self.realized_complexity_of(model, cfg))

        order = np.argsort(comp)
        cfg_sorted = [cfg_list[i] for i in order]
        comp_sorted = comp[order]

        train_sorted = train_err[order]
        test_sorted = test_err[order]
        real_sorted = real_comp[order]
        models_sorted = [models[i] for i in order]

        self.models_ = models_sorted
        best_idx = int(np.argmin(test_sorted))

        self.result_ = ComplexityStudyResult(
            configs=cfg_sorted,
            complexity=comp_sorted,
            train_error=train_sorted,
            test_error=test_sorted,
            realized_complexity=real_sorted,
            best_idx=best_idx,
        )
        return self.result_

    def _run_multi_dataset(
        self,
        cfg_list: list[Config],
        comp: np.ndarray,
    ) -> ComplexityStudyResult:
        n_cfg = len(cfg_list)

        train_curves: list[np.ndarray] = []
        test_curves: list[np.ndarray] = []
        real_curves: list[np.ndarray] = []
        models_per_ds: list[list[Any]] = []

        for X_all, y_all in self.datasets:
            X_tr, X_te, y_tr, y_te = train_test_split(
                X_all,
                y_all,
                test_size=self.test_size,
                random_state=self.random_state,
                stratify=y_all if self.stratify else None,
            )
            X_tr, X_te, _ = self._scale_split(X_tr, X_te)

            train_err = np.empty(n_cfg, dtype=float)
            test_err = np.empty(n_cfg, dtype=float)
            real_comp = np.empty(n_cfg, dtype=float)
            models_this: list[Any] = []

            for i, cfg in enumerate(cfg_list):
                model = self.make_model(cfg)
                model.fit(X_tr, y_tr)
                models_this.append(model)

                yhat_tr = model.predict(X_tr)
                yhat_te = model.predict(X_te)

                score_tr = float(self.scorer(y_tr, yhat_tr))
                score_te = float(self.scorer(y_te, yhat_te))

                train_err[i] = 1.0 - score_tr
                test_err[i] = 1.0 - score_te

                if self.realized_complexity_of is None:
                    real_comp[i] = comp[i]
                else:
                    real_comp[i] = float(self.realized_complexity_of(model, cfg))

            train_curves.append(train_err)
            test_curves.append(test_err)
            real_curves.append(real_comp)
            models_per_ds.append(models_this)

        train_mat = np.vstack(train_curves)  # (n_ds, n_cfg)
        test_mat = np.vstack(test_curves)
        real_mat = np.vstack(real_curves)

        train_agg = np.apply_along_axis(self.aggregate, 0, train_mat)
        test_agg = np.apply_along_axis(self.aggregate, 0, test_mat)
        real_agg = np.apply_along_axis(self.aggregate, 0, real_mat)

        order = np.argsort(comp)
        cfg_sorted = [cfg_list[i] for i in order]
        comp_sorted = comp[order]

        train_sorted = train_agg[order]
        test_sorted = test_agg[order]
        real_sorted = real_agg[order]

        models_sorted = [[m[i] for i in order] for m in models_per_ds]
        train_ds_sorted = train_mat[:, order]
        test_ds_sorted = test_mat[:, order]
        real_ds_sorted = real_mat[:, order]

        self.models_ = {
            "models": models_sorted,
            "train_ds": train_ds_sorted,
            "test_ds": test_ds_sorted,
            "real_ds": real_ds_sorted,
        }

        best_idx = int(np.argmin(test_sorted))
        self.result_ = ComplexityStudyResult(
            configs=cfg_sorted,
            complexity=comp_sorted,
            train_error=train_sorted,
            test_error=test_sorted,
            realized_complexity=real_sorted,
            best_idx=best_idx,
        )
        return self.result_

    @property
    def best_model_(self) -> Any:
        if self.result_ is None:
            raise RuntimeError("Call run(configs) first.")

        if len(self.datasets) == 1:
            return self.models_[self.result_.best_idx]

        models_per_ds = self.models_["models"]
        return [ms[self.result_.best_idx] for ms in models_per_ds]

    @property
    def best_config_(self) -> Config:
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
    plot_mode: Literal["mean", "all", "both"] = "mean",
    ) -> None:
        if self.result_ is None:
            raise RuntimeError("Call run(configs) first.")

        res = self.result_

        if len(self.datasets) == 1 or plot_mode == "mean":
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(res.complexity, res.train_error, marker=marker, label=labels[0])
            ax.plot(res.complexity, res.test_error, marker=marker, label=labels[1])
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
            ax.legend()

            if odd_xticks and np.all(np.isclose(res.complexity, np.round(res.complexity))):
                x_int = res.complexity.astype(int)
                x_odd = x_int[x_int % 2 == 1]
                ax.set_xticks(x_odd)

            plt.show()
            return

        train_ds = self.models_["train_ds"]
        test_ds = self.models_["test_ds"]
        n_ds = int(train_ds.shape[0])

        if plot_mode == "all":
            fig, axs = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
            axs_flat = axs.ravel()

            for j, ax in enumerate(axs_flat[:n_ds]):
                ax.plot(res.complexity, train_ds[j], marker=marker, label=labels[0])
                ax.plot(res.complexity, test_ds[j], marker=marker, label=labels[1])
                ax.set_title(f"{title} (Dataset {j + 1})")
                ax.grid(True, alpha=0.3)
                ax.legend()

                if odd_xticks and np.all(
                    np.isclose(res.complexity, np.round(res.complexity))
                ):
                    x_int = res.complexity.astype(int)
                    x_odd = x_int[x_int % 2 == 1]
                    ax.set_xticks(x_odd)

            for ax in axs_flat:
                ax.set_xlabel(xlabel)
                ax.set_ylabel(ylabel)

            plt.tight_layout()
            plt.show()
            return

        # plot_mode == "both": 4 subplots + 1 mean plot
        fig, axs = plt.subplots(
            3,
            2,
            figsize=(12, 12),
            sharex=True,
            sharey=True,
        )
        axs_flat = axs.ravel()

        # First 4 panels: per-dataset
        for j in range(min(4, n_ds)):
            ax = axs_flat[j]
            ax.plot(res.complexity, train_ds[j], marker=marker, label=labels[0])
            ax.plot(res.complexity, test_ds[j], marker=marker, label=labels[1])
            ax.set_title(f"{title} (Dataset {j + 1})")
            ax.grid(True, alpha=0.3)
            ax.legend()

            if odd_xticks and np.all(np.isclose(res.complexity, np.round(res.complexity))):
                x_int = res.complexity.astype(int)
                x_odd = x_int[x_int % 2 == 1]
                ax.set_xticks(x_odd)

        ax_mean = axs_flat[4]
        ax_mean.plot(res.complexity, res.train_error, marker=marker, label=labels[0])
        ax_mean.plot(res.complexity, res.test_error, marker=marker, label=labels[1])
        ax_mean.set_title(f"{title} (Mean)")
        ax_mean.grid(True, alpha=0.3)
        ax_mean.legend()

        if odd_xticks and np.all(np.isclose(res.complexity, np.round(res.complexity))):
            x_int = res.complexity.astype(int)
            x_odd = x_int[x_int % 2 == 1]
            ax_mean.set_xticks(x_odd)

        # Sixth panel unused
        axs_flat[5].axis("off")

        for ax in axs_flat[:5]:
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)

        plt.tight_layout()
        plt.show()


    def plot_realized_complexity(
        self,
        *,
        title: str,
        xlabel: str,
        ylabel: str,
        marker: str = "o",
        odd_xticks: bool = False,
        plot_mode: Literal["mean", "all", "both"] = "mean",
        ) -> None:
            if self.result_ is None:
                raise RuntimeError("Call run(configs) first.")

            res = self.result_

            if len(self.datasets) == 1 or plot_mode == "mean":
                fig, ax = plt.subplots(figsize=(7, 4))
                ax.plot(res.complexity, res.realized_complexity, marker=marker, label="Realized")
                ax.set_xlabel(xlabel)
                ax.set_ylabel(ylabel)
                ax.set_title(title)
                ax.grid(True, alpha=0.3)
                ax.legend()

                if odd_xticks and np.all(np.isclose(res.complexity, np.round(res.complexity))):
                    x_int = res.complexity.astype(int)
                    x_odd = x_int[x_int % 2 == 1]
                    ax.set_xticks(x_odd)

                plt.show()
                return

            real_ds = self.models_["real_ds"]
            n_ds = int(real_ds.shape[0])

            if plot_mode == "all":
                fig, axs = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
                axs_flat = axs.ravel()

                for j, ax in enumerate(axs_flat[:n_ds]):
                    ax.plot(
                        res.complexity,
                        real_ds[j],
                        marker=marker,
                        label="Realized",
                    )
                    ax.set_title(f"{title} (Dataset {j + 1})")
                    ax.grid(True, alpha=0.3)
                    ax.legend()

                    if odd_xticks and np.all(
                        np.isclose(res.complexity, np.round(res.complexity))
                    ):
                        x_int = res.complexity.astype(int)
                        x_odd = x_int[x_int % 2 == 1]
                        ax.set_xticks(x_odd)

                for ax in axs_flat:
                    ax.set_xlabel(xlabel)
                    ax.set_ylabel(ylabel)

                plt.tight_layout()
                plt.show()
                return

            # plot_mode == "both": 4 subplots + 1 mean plot
            fig, axs = plt.subplots(
                3,
                2,
                figsize=(12, 12),
                sharex=True,
                sharey=True,
            )
            axs_flat = axs.ravel()

            for j in range(min(4, n_ds)):
                ax = axs_flat[j]
                ax.plot(
                    res.complexity,
                    real_ds[j],
                    marker=marker,
                    label="Realized",
                )
                ax.set_title(f"{title} (Dataset {j + 1})")
                ax.grid(True, alpha=0.3)
                ax.legend()

                if odd_xticks and np.all(np.isclose(res.complexity, np.round(res.complexity))):
                    x_int = res.complexity.astype(int)
                    x_odd = x_int[x_int % 2 == 1]
                    ax.set_xticks(x_odd)

            ax_mean = axs_flat[4]
            ax_mean.plot(
                res.complexity,
                res.realized_complexity,
                marker=marker,
                label="Realized (Mean)",
            )
            ax_mean.set_title(f"{title} (Mean)")
            ax_mean.grid(True, alpha=0.3)
            ax_mean.legend()

            if odd_xticks and np.all(np.isclose(res.complexity, np.round(res.complexity))):
                x_int = res.complexity.astype(int)
                x_odd = x_int[x_int % 2 == 1]
                ax_mean.set_xticks(x_odd)

            axs_flat[5].axis("off")

            for ax in axs_flat[:5]:
                ax.set_xlabel(xlabel)
                ax.set_ylabel(ylabel)

            plt.tight_layout()
            plt.show()