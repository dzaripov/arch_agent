"""Example showing how to launch an Informer2020 Optuna search programmatically."""
from __future__ import annotations

from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

from optuna_module import (
    InformerSearchSpace,
    start_informer2020_optuna_search,
    suggest_start_informer2020_search_kwargs,
)


def main() -> None:
    """Start a short Informer2020 search with hard-coded parameters."""

    custom_search_space = InformerSearchSpace(
        seq_len_choices=(96, 168, 336, 512),
        label_len_choices=(48, 96, 168, 256),
        pred_len_choices=(24, 96, 168, 336, 512),
        d_model_choices=(256, 512, 768),
        n_heads_choices=(4, 8, 16),
        e_layers_choices=(2, 3, 4),
        d_layers_choices=(1, 2, 3),
        d_ff_choices=(512, 1024, 2048, 4096),
        factor_choices=(1, 3, 5, 7),
        dropout=(0.01, 0.4),
        learning_rate=(5e-6, 1e-3),
        batch_size_choices=(16, 32, 64, 128),
        train_epochs=(4, 24),
        patience=(2, 8),
        s_layers_choices=("4,3,2", "5,4,3", "3,2,1"),
        attn_choices=("prob", "full"),
        embed_choices=("timeF", "fixed", "learned"),
        activation_choices=("gelu", "relu"),
        distil_options=(True, False),
        output_attention_options=(False, True),
        mix_options=(True, False),
        padding_options=(0, 1),
        lradj_choices=("type1", "type2"),
    )

    kwargs = suggest_start_informer2020_search_kwargs(
        n_trials=5,
        metric="val_loss",
        seed=42,
        fixed_parameters={
            "train_epochs": 6,
            "batch_size": 32,
        },
        search_space=custom_search_space,
        sampler=TPESampler(seed=42),
        pruner=MedianPruner(n_startup_trials=2, n_warmup_steps=0, interval_steps=1),
    )

    study = start_informer2020_optuna_search(**kwargs)
    print(f"Best trial value: {study.best_value}")


if __name__ == "__main__":
    main()
