
from __future__ import annotations

from collections.abc import Sequence as ABCSequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Sequence

import argparse
import json
import sys

import optuna
from optuna import Trial
from optuna.pruners import BasePruner, MedianPruner
from optuna.samplers import BaseSampler, TPESampler
from optuna.visualization import plot_contour, plot_optimization_history
from optuna.visualization import plot_parallel_coordinate, plot_param_importances
from optuna.visualization import plot_slice


@dataclass(slots=True)
class InformerSearchSpace:
    seq_len_choices: Sequence[int] = (96,)
    label_len_choices: Sequence[int] = (48,)
    pred_len_choices: Sequence[int] = (24,)
    d_model_choices: Sequence[int] = (256, 512)
    n_heads_choices: Sequence[int] = (4, 8)
    e_layers_choices: Sequence[int] = (2,)
    d_layers_choices: Sequence[int] = (1,)
    d_ff_choices: Sequence[int] = (512, 1024)
    factor_choices: Sequence[int] = (1, 3)
    dropout: tuple[float, float] = (0.01, 0.1)
    learning_rate: tuple[float, float] = (1e-5, 2e-4)
    batch_size_choices: Sequence[int] = (16, 32)
    train_epochs: tuple[int, int] = (4, 12)
    patience: tuple[int, int] = (2, 4)
    s_layers_choices: Sequence[str] = ("3,2,1",)
    attn_choices: Sequence[str] = ("prob", "full")
    embed_choices: Sequence[str] = ("timeF", "fixed")
    activation_choices: Sequence[str] = ("gelu",)
    distil_options: Sequence[bool] = (True, False)
    output_attention_options: Sequence[bool] = (False, True)
    mix_options: Sequence[bool] = (True, False)
    padding_options: Sequence[int] = (0,)
    lradj_choices: Sequence[str] = ("type1",)


def suggest_informer_hyperparameters(
    trial: Trial,
    search_space: InformerSearchSpace | None = None,
) -> Dict[str, Any]:

    space = search_space or InformerSearchSpace()

    seq_len = trial.suggest_categorical("seq_len", list(space.seq_len_choices))
    label_len = trial.suggest_categorical("label_len", list(space.label_len_choices))
    if label_len > seq_len:
        raise optuna.TrialPruned("label_len cannot exceed seq_len")
    pred_len = trial.suggest_categorical("pred_len", list(space.pred_len_choices))

    pred_len = trial.suggest_categorical("pred_len", list(space.pred_len_choices))

    params: Dict[str, Any] = {
        "seq_len": seq_len,
        "label_len": label_len,
        "pred_len": pred_len,
        "d_model": trial.suggest_categorical("d_model", list(space.d_model_choices)),
        "n_heads": trial.suggest_categorical("n_heads", list(space.n_heads_choices)),
        "e_layers": trial.suggest_categorical("e_layers", list(space.e_layers_choices)),
        "d_layers": trial.suggest_categorical("d_layers", list(space.d_layers_choices)),
        "d_ff": trial.suggest_categorical("d_ff", list(space.d_ff_choices)),
        "factor": trial.suggest_categorical("factor", list(space.factor_choices)),
        "dropout": trial.suggest_float("dropout", *space.dropout),
        "learning_rate": trial.suggest_float(
            "learning_rate", *space.learning_rate, log=True
        ),
        "batch_size": trial.suggest_categorical("batch_size", list(space.batch_size_choices)),
        "train_epochs": trial.suggest_int("train_epochs", *space.train_epochs),
        "patience": trial.suggest_int("patience", *space.patience),
        "s_layers": trial.suggest_categorical("s_layers", list(space.s_layers_choices)),
        "attn": trial.suggest_categorical("attn", list(space.attn_choices)),
        "embed": trial.suggest_categorical("embed", list(space.embed_choices)),
        "activation": trial.suggest_categorical("activation", list(space.activation_choices)),
        "distil": trial.suggest_categorical("distil", list(space.distil_options)),
        "output_attention": trial.suggest_categorical(
            "output_attention", list(space.output_attention_options)
        ),
        "mix": trial.suggest_categorical("mix", list(space.mix_options)),
        "padding": trial.suggest_categorical("padding", list(space.padding_options)),
        "lradj": trial.suggest_categorical("lradj", list(space.lradj_choices)),
    }

    return params


def _extract_metric_value(metric: Any) -> float:

    if isinstance(metric, (int, float)):
        return float(metric)

    if isinstance(metric, ABCSequence) and not isinstance(metric, (str, bytes, bytearray)):
        if not metric:
            raise ValueError("Metric sequence is empty; cannot derive an objective value.")
        return float(metric[-1])

    raise TypeError(
        "Unsupported metric value type. Expected a numeric value or a sequence of numbers."
    )


def create_objective(
    experiment_runner: Callable[..., Mapping[str, Any]],
    *,
    metric: str = "val_loss",
    greater_is_better: bool = False,
    search_space: InformerSearchSpace | None = None,
    fixed_parameters: Optional[Mapping[str, Any]] = None,
) -> Callable[[Trial], float]:

    fixed = dict(fixed_parameters or {})

    def objective(trial: Trial) -> float:
        sampled_params = suggest_informer_hyperparameters(trial, search_space)
        sampled_params.update(fixed)

        result = experiment_runner(**sampled_params)
        if metric not in result:
            raise KeyError(
                f"Experiment runner did not return the requested metric '{metric}'."
            )

        metric_value = _extract_metric_value(result[metric])

        trial.set_user_attr("metrics", result)

        return -metric_value if greater_is_better else metric_value

    return objective


def run_study(
    n_trials: int,
    *,
    experiment_runner: Callable[..., Mapping[str, Any]],
    metric: str = "val_loss",
    greater_is_better: bool = False,
    search_space: InformerSearchSpace | None = None,
    fixed_parameters: Optional[Mapping[str, Any]] = None,
    study: optuna.Study | None = None,
    sampler: BaseSampler | None = None,
    pruner: BasePruner | None = None,
    study_name: str | None = None,
    storage: str | None = None,
    load_if_exists: bool = True,
    **optimize_kwargs: Any,
) -> optuna.Study:

    if n_trials <= 0:
        raise ValueError("n_trials must be a positive integer.")

    direction = "maximize" if greater_is_better else "minimize"
    if study is None:
        create_kwargs: Dict[str, Any] = {"direction": direction}
        if sampler is not None:
            create_kwargs["sampler"] = sampler
        if pruner is not None:
            create_kwargs["pruner"] = pruner
        if study_name is not None:
            create_kwargs["study_name"] = study_name
        if storage is not None:
            create_kwargs["storage"] = storage
            create_kwargs["load_if_exists"] = load_if_exists

        study = optuna.create_study(**create_kwargs)

    objective = create_objective(
        experiment_runner,
        metric=metric,
        greater_is_better=greater_is_better,
        search_space=search_space,
        fixed_parameters=fixed_parameters,
    )

    study.optimize(objective, n_trials=n_trials, **optimize_kwargs)
    return study


def _load_default_informer_experiment_runner() -> Callable[..., Mapping[str, Any]]:
    try:
        from informer_tool import run_informer_experiment  # type: ignore
    except ModuleNotFoundError:
        module_dir = Path(__file__).resolve().parent
        informer_dir = module_dir / "Informer2020"

        search_paths = [str(module_dir)]
        if informer_dir.exists():
            search_paths.insert(0, str(informer_dir))

        for path in search_paths:
            if path not in sys.path:
                sys.path.insert(0, path)

        try:
            from informer_tool import run_informer_experiment  # type: ignore
        except ModuleNotFoundError as import_error:
            raise ImportError(
                "Unable to import 'run_informer_experiment'. Ensure the Informer2020 "
                "sources are available next to optuna_module.py."
            ) from import_error

    return run_informer_experiment


def start_informer2020_optuna_search(
    n_trials: int,
    *,
    metric: str = "val_loss",
    greater_is_better: bool = False,
    search_space: InformerSearchSpace | None = None,
    fixed_parameters: Optional[Mapping[str, Any]] = None,
    sampler: BaseSampler | None = None,
    pruner: BasePruner | None = None,
    use_default_sampler: bool = True,
    use_default_pruner: bool = True,
    study_name: str | None = None,
    storage: str | None = None,
    load_if_exists: bool = True,
    experiment_runner: Callable[..., Mapping[str, Any]] | None = None,
    show_progress_bar: bool = True,
    **optimize_kwargs: Any,
) -> optuna.Study:
    if sampler is None and use_default_sampler:
        sampler = TPESampler()
    if pruner is None and use_default_pruner:
        pruner = MedianPruner(n_startup_trials=2, n_warmup_steps=0, interval_steps=1)

    if experiment_runner is None:
        experiment_runner = _load_default_informer_experiment_runner()

    return run_study(
        n_trials,
        experiment_runner=experiment_runner,
        metric=metric,
        greater_is_better=greater_is_better,
        search_space=search_space,
        fixed_parameters=fixed_parameters,
        sampler=sampler,
        pruner=pruner,
        study_name=study_name,
        storage=storage,
        load_if_exists=load_if_exists,
        show_progress_bar=show_progress_bar,
        **optimize_kwargs,
    )


def _create_sampler(seed: Optional[int]) -> BaseSampler:
    return TPESampler(seed=seed)


def _create_pruner(disabled: bool) -> BasePruner | None:
    if disabled:
        return None
    return MedianPruner(n_startup_trials=2, n_warmup_steps=0, interval_steps=1)


def suggest_start_informer2020_search_kwargs(
    *,
    n_trials: int,
    metric: str = "val_loss",
    greater_is_better: bool = False,
    seed: Optional[int] = None,
    disable_pruner: bool = False,
    fixed_parameters: Optional[Mapping[str, Any]] = None,
    study_name: str | None = None,
    storage: str | None = None,
    force_new_study: bool = False,
    show_progress_bar: bool = True,
    n_jobs: int = 1,
    timeout: float | None = None,
    search_space: InformerSearchSpace | None = None,
    sampler: BaseSampler | None = None,
    pruner: BasePruner | None = None,
) -> Dict[str, Any]:


    use_default_sampler = False
    effective_sampler = sampler
    if effective_sampler is None:
        if seed is None:
            use_default_sampler = True
        else:
            effective_sampler = _create_sampler(seed)

    if disable_pruner:
        effective_pruner: BasePruner | None = None
        use_default_pruner = False
    elif pruner is not None:
        effective_pruner = pruner
        use_default_pruner = False
    else:
        effective_pruner = None
        use_default_pruner = True

    return {
        "n_trials": n_trials,
        "metric": metric,
        "greater_is_better": greater_is_better,
        "sampler": effective_sampler,
        "pruner": effective_pruner,
        "use_default_sampler": use_default_sampler,
        "use_default_pruner": use_default_pruner,
        "study_name": study_name,
        "storage": storage,
        "fixed_parameters": dict(fixed_parameters) if fixed_parameters is not None else None,
        "load_if_exists": not force_new_study,
        "show_progress_bar": show_progress_bar,
        "n_jobs": n_jobs,
        "timeout": timeout,
        "search_space": search_space,
    }


def _print_study_summary(study: optuna.Study) -> None:
    best_trial = study.best_trial
    print("\nBest trial:")
    print(f"  Number: {best_trial.number}")
    print(f"  Value: {best_trial.value}")
    print("  Parameters:")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")

    metrics = best_trial.user_attrs.get("metrics")
    if isinstance(metrics, Mapping):
        print("  Metrics:")
        print(json.dumps(metrics, indent=2))


def save_study_visualizations(
    study: optuna.Study,
    *,
    output_dir: str | Path = "optuna_plots",
    contour_params: Sequence[str] | None = None,
) -> list[Path]:

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    figures = {
        "optimization_history.html": plot_optimization_history(study),
        "param_importances.html": plot_param_importances(study),
        "parallel_coordinate.html": plot_parallel_coordinate(study),
        "contour.html": plot_contour(study, params=list(contour_params) if contour_params else None),
        "slice.html": plot_slice(study),
    }

    saved_paths: list[Path] = []
    for filename, figure in figures.items():
        file_path = output_path / filename
        figure.write_html(file_path, include_plotlyjs="cdn")
        saved_paths.append(file_path)

    return saved_paths


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    sampler = _create_sampler(args.seed)
    pruner = _create_pruner(args.disable_pruner)

    fixed_parameters: Optional[Mapping[str, Any]]
    if args.fixed_params is None:
        fixed_parameters = None
    elif isinstance(args.fixed_params, Mapping):
        fixed_parameters = dict(args.fixed_params)
    else:
        parser.error("--fixed-params must decode to a JSON object (mapping).")
        return

    search_kwargs = suggest_start_informer2020_search_kwargs(
        n_trials=args.n_trials,
        metric=args.metric,
        greater_is_better=args.greater_is_better,
        seed=args.seed,
        disable_pruner=args.disable_pruner,
        fixed_parameters=fixed_parameters,
        study_name=args.study_name,
        storage=args.storage,
        force_new_study=args.force_new_study,
        show_progress_bar=not args.no_progress_bar,
        n_jobs=args.n_jobs,
        timeout=args.timeout,
        search_space=None,
        sampler=sampler,
        pruner=pruner,
    )

    study = start_informer2020_optuna_search(**search_kwargs)

    _print_study_summary(study)


if __name__ == "__main__":
    main()


__all__ = [
    "InformerSearchSpace",
    "suggest_informer_hyperparameters",
    "create_objective",
    "run_study",
    "start_informer2020_optuna_search",
    "suggest_start_informer2020_search_kwargs",
    "save_study_visualizations",
]
