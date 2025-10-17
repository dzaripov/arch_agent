import sys
import os

# print(os.path.join(os.getcwd(), "./Informer2020/"))
sys.path.append(os.path.join(os.getcwd(), "./Informer2020/"))


from Informer2020.exp.exp_informer import Exp_Informer

informer_tools = [
    {
        "type": "function",
        "function": {
            "name": "run_informer_experiment",
            "description": "Runs a long sequence time-series forecasting experiment using the Informer model.",
            "parameters": {
                "type": "object",
                "properties": {
                    "seq_len": {
                        "type": "integer",
                        "description": "Input sequence length for the Informer encoder. Default is 96.",
                        "default": 96,
                    },
                    "label_len": {
                        "type": "integer",
                        "description": "Start token length for the Informer decoder. Default is 48.",
                        "default": 48,
                    },
                    "pred_len": {
                        "type": "integer",
                        "description": "Prediction sequence length. Default is 24.",
                        "default": 24,
                    },
                    "d_model": {
                        "type": "integer",
                        "description": "Dimension of the model. Default is 512.",
                        "default": 512,
                    },
                    "n_heads": {
                        "type": "integer",
                        "description": "Number of heads in the multi-head attention mechanism. Default is 8.",
                        "default": 8,
                    },
                    "e_layers": {
                        "type": "integer",
                        "description": "Number of encoder layers. Default is 2.",
                        "default": 2,
                    },
                    "d_layers": {
                        "type": "integer",
                        "description": "Number of decoder layers. Default is 1.",
                        "default": 1,
                    },
                    "s_layers": {
                        "type": "string",
                        "description": "Number of stack encoder layers, formatted as a comma-separated string (e.g., '3,2,1'). Default is '3,2,1'.",
                        "default": "3,2,1",
                    },
                    "d_ff": {
                        "type": "integer",
                        "description": "Dimension of the fully connected network. Default is 2048.",
                        "default": 2048,
                    },
                    "factor": {
                        "type": "integer",
                        "description": "ProbSparse attention factor. Default is 5.",
                        "default": 5,
                    },
                    "padding": {
                        "type": "integer",
                        "description": "Padding type. Use 0 for zero padding and 1 for border padding. Default is 0.",
                        "enum": [0, 1],
                        "default": 0,
                    },
                    "distil": {
                        "type": "boolean",
                        "description": "Whether to use distilling in the encoder. Default is True.",
                        "default": True,
                    },
                    "dropout": {
                        "type": "number",
                        "description": "Dropout rate. Default is 0.05.",
                        "default": 0.05,
                    },
                    "attn": {
                        "type": "string",
                        "description": "Attention mechanism used in the encoder.",
                        "enum": ["prob", "full"],
                        "default": "prob",
                    },
                    "embed": {
                        "type": "string",
                        "description": "Time features encoding type.",
                        "enum": ["timeF", "fixed", "learned"],
                        "default": "timeF",
                    },
                    "activation": {
                        "type": "string",
                        "description": "Activation function. Default is 'gelu'.",
                        "default": "gelu",
                    },
                    "output_attention": {
                        "type": "boolean",
                        "description": "Whether to output attention scores from the encoder. Default is False.",
                        "default": False,
                    },
                    "mix": {
                        "type": "boolean",
                        "description": "Whether to use mix attention in the generative decoder. Default is True.",
                        "default": True,
                    },
                    "train_epochs": {
                        "type": "integer",
                        "description": "Number of training epochs. Default is 6.",
                        "default": 6,
                    },
                    "batch_size": {
                        "type": "integer",
                        "description": "Batch size for training input data. Default is 32.",
                        "default": 32,
                    },
                    "patience": {
                        "type": "integer",
                        "description": "Patience for early stopping. Default is 3.",
                        "default": 3,
                    },
                    "learning_rate": {
                        "type": "number",
                        "description": "Optimizer learning rate. Default is 0.0001.",
                        "default": 0.0001,
                    },
                    "lradj": {
                        "type": "string",
                        "description": "Learning rate adjustment strategy. Default is 'type1'.",
                        "enum": ['type1', 'type2'],
                        "default": "type1",
                    },
                },
                "required": [], 
            },
        },
    }
]

import argparse
import os
import torch
import numpy as np
import warnings

warnings.filterwarnings("ignore")


def run_informer_experiment(
    seq_len: int = 96,
    label_len: int = 48,
    pred_len: int = 24,
    d_model: int = 512,
    n_heads: int = 8,
    e_layers: int = 2,
    d_layers: int = 1,
    s_layers: str = "3,2,1",
    d_ff: int = 2048,
    factor: int = 5,
    padding: int = 0,
    distil: bool = True,
    dropout: float = 0.05,
    attn: str = "prob",
    embed: str = "timeF",
    activation: str = "gelu",
    output_attention: bool = False,
    mix: bool = True,
    train_epochs: int = 6,
    batch_size: int = 32,
    patience: int = 3,
    learning_rate: float = 0.0001,
    lradj: str = "type1",
):
    """
    Runs a long sequence time-series forecasting experiment using the Informer model.

    This function encapsulates the logic from the original main_informer.py script,
    allowing it to be called programmatically as a tool.

    Args:
        seq_len (int): Input sequence length for the Informer encoder.
        label_len (int): Start token length for the Informer decoder.
        pred_len (int): Prediction sequence length.
        d_model (int): Dimension of the model.
        n_heads (int): Number of heads.
        e_layers (int): Number of encoder layers.
        d_layers (int): Number of decoder layers.
        s_layers (str): Comma-separated string of stack encoder layers.
        d_ff (int): Dimension of the feed-forward network.
        factor (int): ProbSparse attention factor.
        padding (int): Padding type.
        distil (bool): Whether to use distilling in the encoder.
        dropout (float): Dropout rate.
        attn (str): Attention mechanism used in the encoder ('prob' or 'full').
        embed (str): Time features encoding type.
        activation (str): Activation function.
        output_attention (bool): Whether to output attention in the encoder.
        mix (bool): Whether to use mix attention in the generative decoder.
        train_epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
        patience (int): Early stopping patience.
        learning_rate (float): Optimizer learning rate.
        lradj (str): Learning rate adjustment strategy.

    Returns:
        dict: A dictionary containing the final test metrics (epochs, train_loss, val_loss)
              for each iteration of the experiment.
    """
    # Create an args object to mimic the original script's argument parsing
    args = argparse.Namespace()

    # test
    # train_epochs = 1

    # not hyperparameters
    model = 'informer'
    data = 'ETTh1'
    root_path = "./Informer2020/data/ETT/"
    features = "M"
    freq = "h"
    checkpoints = "./Informer2020/checkpoints/"
    do_predict = False
    use_amp = False
    inverse = False
    use_gpu = True
    gpu = 0
    use_multi_gpu = False
    devices = "0,1,2,3"
    cols = None
    itr = 1
    num_workers = 0
    des = "run"
    loss = 'mse' # the only loss in the code



    
    # Assign all function parameters to the args object
    args.model, args.data, args.root_path, args.features, args.freq = (
        model,
        data,
        root_path,
        features,
        freq,
    )
    args.checkpoints, args.seq_len, args.label_len, args.pred_len = (
        checkpoints,
        seq_len,
        label_len,
        pred_len,
    )
    args.d_model, args.n_heads, args.e_layers, args.d_layers, args.s_layers = (
        d_model,
        n_heads,
        e_layers,
        d_layers,
        s_layers,
    )
    args.d_ff, args.factor, args.padding, args.distil, args.dropout = (
        d_ff,
        factor,
        padding,
        distil,
        dropout,
    )
    args.attn, args.embed, args.activation, args.output_attention = (
        attn,
        embed,
        activation,
        output_attention,
    )
    args.do_predict, args.mix, args.cols, args.num_workers, args.itr = (
        do_predict,
        mix,
        cols,
        num_workers,
        itr,
    )
    args.train_epochs, args.batch_size, args.patience, args.learning_rate = (
        train_epochs,
        batch_size,
        patience,
        learning_rate,
    )
    args.des, args.loss, args.lradj, args.use_amp, args.inverse = (
        des,
        loss,
        lradj,
        use_amp,
        inverse,
    )
    args.use_gpu, args.gpu, args.use_multi_gpu, args.devices = (
        use_gpu,
        gpu,
        use_multi_gpu,
        devices,
    )

    # --- Replicate setup from main_informer.py ---

    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(" ", "")
        device_ids = args.devices.split(",")
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]
    else:
        # Ensure multi-gpu is disabled if not available or not requested
        args.use_multi_gpu = False

    data_parser = {
        "ETTh1": {
            "data": "ETTh1.csv",
            "T": "OT",
            "M": [7, 7, 7],
            "S": [1, 1, 1],
            "MS": [7, 7, 1],
        },
        "ETTh2": {
            "data": "ETTh2.csv",
            "T": "OT",
            "M": [7, 7, 7],
            "S": [1, 1, 1],
            "MS": [7, 7, 1],
        },
        "ETTm1": {
            "data": "ETTm1.csv",
            "T": "OT",
            "M": [7, 7, 7],
            "S": [1, 1, 1],
            "MS": [7, 7, 1],
        },
        "ETTm2": {
            "data": "ETTm2.csv",
            "T": "OT",
            "M": [7, 7, 7],
            "S": [1, 1, 1],
            "MS": [7, 7, 1],
        },
        "WTH": {
            "data": "WTH.csv",
            "T": "WetBulbCelsius",
            "M": [12, 12, 12],
            "S": [1, 1, 1],
            "MS": [12, 12, 1],
        },
        "ECL": {
            "data": "ECL.csv",
            "T": "MT_320",
            "M": [321, 321, 321],
            "S": [1, 1, 1],
            "MS": [321, 321, 1],
        },
        "Solar": {
            "data": "solar_AL.csv",
            "T": "POWER_136",
            "M": [137, 137, 137],
            "S": [1, 1, 1],
            "MS": [137, 137, 1],
        },
    }
    if args.data in data_parser.keys():
        data_info = data_parser[args.data]
        args.data_path = data_info["data"]
        args.target = data_info["T"]
        args.enc_in, args.dec_in, args.c_out = data_info[args.features]

    # These will be set automatically based on the dataset and features choice
    args.s_layers = [int(s_l) for s_l in args.s_layers.replace(" ", "").split(",")]
    args.detail_freq = args.freq
    args.freq = args.freq[-1:]

    Exp = Exp_Informer

    # iterations are not used now!
    for ii in range(args.itr):
        setting = "{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_at{}_fc{}_eb{}_dt{}_mx{}_{}_{}".format(
            args.model,
            args.data,
            args.features,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.d_model,
            args.n_heads,
            args.e_layers,
            args.d_layers,
            args.d_ff,
            args.attn,
            args.factor,
            args.embed,
            args.distil,
            args.mix,
            args.des,
            ii,
        )

        exp = Exp(args)
        res = exp.train(setting)

        # save everything in res to use for analytics and test_loss

        torch.cuda.empty_cache()

    return {'epochs': res["epochs"],
            'train_loss': res["train_loss"],
            'train_mae': res['train_mae'],
            'val_loss': res["val_loss"],
            'val_mae': res['val_mae']}


# Example of how to call the function
if __name__ == "__main__":
    # This is a demonstration call.
    if not os.path.exists("./Informer2020/data/ETT/ETTh1.csv"):
        print("Dataset ETTh1.csv not found. Skipping example run.")
    else:
        experiment_results = run_informer_experiment(
            seq_len=96,
            label_len=48,
            pred_len=24,
            e_layers=2,
            d_layers=1,
            attn="prob",
            train_epochs=4,
        )
        print("\n--- Experiment Finished ---")
        print("Results:")
        import json

        print(json.dumps(experiment_results, indent=2))
