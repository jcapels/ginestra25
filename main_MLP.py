import os, time
import torch
import torch.nn as nn
import torch.optim as optim


from utils.earlystop import EarlyStopping
from utils.utils import initialize_experiment, final_stats
from utils.epoch_functions import training_epoch, evaluation_epoch

from config import GRID_N_EPOCHS, N_RUNS, LABELS_CODES, TARGET_TYPE, BASEDIR, DATASET_ID, EARLY_PATIENCE, EARLY_MIN_DELTA, USE_MULTILABEL, DEVICE as device

from models.MLP import *

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

from gridsearch_dataset_builder import prepare_dataloaders

MODEL_NAME = "mlp"

train_dataloader, val_dataloader, test_dataloader = prepare_dataloaders(MODEL_NAME)

EXPERIMENT_FOLDER = initialize_experiment(f"{MODEL_NAME}_{DATASET_ID}", TARGET_TYPE, BASEDIR)

def run_experiment(grid_config, train_loader, val_loader, test_loader, num_classes, config_idx, n_config):
    curr_config_report_file = os.path.join(EXPERIMENT_FOLDER, "reports", f"report_{MODEL_NAME}_{config_idx}.txt")
    grid_statistics = {
        'train_loss': [],
        'train_precision': [],
        'train_recall': [],
        'train_f1': [],
        'val_loss': [],
        'val_precision': [],
        'val_recall': [],
        'val_f1': [],
        'epoch_time': []
    }

    for run in range(N_RUNS):
        
        model = MLP(
            unit1=grid_config['unit1'],
            unit2=grid_config['unit2'],
            unit3=grid_config['unit3'],
            drop_rate=grid_config['drop_rate'],
            num_classes=num_classes
        ).to(device)
        # Reset the model weights
        for layer in model.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        print(model)
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Number of parameters: {num_params}")
        print(f"Model summary: {model}")
        print(f"Model initialized with config: {grid_config}")
        with open(curr_config_report_file, "a") as f:
            f.write(f"Configuration {config_idx}/{n_config} - Run {run+1}/{N_RUNS}\n")
            f.write(f"Number of parameters: {num_params}\n")
            f.write(f"Model summary: {model}\n")
            f.write(f"Model configuration: {grid_config}\n")

        optimizer = optim.AdamW(model.parameters(), lr=grid_config['learning_rate'], weight_decay=grid_config['l2_rate'])
        criterion = nn.CrossEntropyLoss() if not USE_MULTILABEL else nn.BCEWithLogitsLoss()

        early_stopping = EarlyStopping(
            patience=EARLY_PATIENCE,
            min_delta=EARLY_MIN_DELTA,
            verbose=True,
            path=os.path.join(EXPERIMENT_FOLDER, "models", f"best_model_config_{config_idx}_run_{run+1}.pt"),
            metric_name="val_f1", 
            grid_config=grid_config
            )

        for epoch in range(GRID_N_EPOCHS):
            print(f"[CONFIG {config_idx}/{n_config}][{MODEL_NAME.upper()} RUN {run+1}/{N_RUNS}] Epoch {epoch+1}/{GRID_N_EPOCHS}")
            start_time = time.time()
            train_loss, train_precision, train_recall, train_f1 = training_epoch(model, train_loader, optimizer, criterion, device, experim_folder=EXPERIMENT_FOLDER)
            end_time = time.time()
            val_loss, val_precision, val_recall, val_f1, topk = evaluation_epoch(model, val_loader, criterion, device, experim_folder=EXPERIMENT_FOLDER)

            grid_statistics['train_loss'].append(train_loss)
            grid_statistics['train_precision'].append(train_precision)
            grid_statistics['train_recall'].append(train_recall)
            grid_statistics['train_f1'].append(train_f1)
            grid_statistics['val_loss'].append(val_loss)
            grid_statistics['val_precision'].append(val_precision)
            grid_statistics['val_recall'].append(val_recall)
            grid_statistics['val_f1'].append(val_f1)
            grid_statistics['epoch_time'].append(end_time - start_time)
            
            # EarlyStopping step
            early_stopping(val_f1, model, epoch)
            if early_stopping.early_stop:
                print(f"Stopped early at epoch {epoch}. Loading best model from {early_stopping.path}")
                model.load_state_dict(torch.load(early_stopping.path))
                test_loss, test_precision, test_recall, test_f1, test_topk = evaluation_epoch(model, test_loader, criterion, device, mode="explain_test")
                final_stats(grid_statistics, config_idx, n_config, early_stopping.last_checkpoint_epoch)
                break

    return grid_statistics

if __name__ == "__main__":
    train_dataloader = train_dataloader
    val_dataloader = val_dataloader
    test_dataloader = test_dataloader
    sample = next(iter(train_dataloader))
    num_classes = len(LABELS_CODES)

    # Esempio di grid search manuale
    from itertools import product
    from config import PARAM_GRID_MLP
    param_grid = PARAM_GRID_MLP
    keys, values = zip(*param_grid.items())
    configs = [dict(zip(keys, v)) for v in product(*values)]
    n_config = len(configs)

    for config_idx, grid_config in enumerate(configs, 1):
        print(f"\n=== CONFIG {config_idx}/{n_config} ===")
        run_experiment(grid_config, train_dataloader, val_dataloader, test_dataloader, num_classes, config_idx, n_config)