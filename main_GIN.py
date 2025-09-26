import os, time
import torch
import torch.nn as nn
import torch.optim as optim


from utils.earlystop import EarlyStopping
from utils.utils import initialize_experiment, final_stats
from utils.epoch_functions import training_epoch, evaluation_epoch

from config import USE_FINGERPRINT, GRID_N_EPOCHS, N_RUNS, TARGET_TYPE, BASEDIR, DATASET_ID, EARLY_PATIENCE, EARLY_MIN_DELTA, USE_MULTILABEL, DEVICE as device

from models.GIN import *

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

from gridsearch_dataset_builder import prepare_dataloaders

MODEL_NAME = "gin"

EXPERIMENT_FOLDER = initialize_experiment(f"{MODEL_NAME}_{DATASET_ID}", TARGET_TYPE, BASEDIR)

def run_experiment(grid_config, train_loader, val_loader, test_loader, num_node_features, edge_dim, num_classes, config_idx, n_config):
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
        
        model = GIN(
            num_node_features=num_node_features,
            dim_h=grid_config['dim_h'],
            num_classes=num_classes,
            drop_rate=grid_config['drop_rate'],
            edge_dim=edge_dim,
            fingerprint=USE_FINGERPRINT,
            num_layers=4
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
            val_loss, val_precision, val_recall,  val_f1, val_precision_pathway, val_recall_pathway, val_f1_pathway, val_precision_superpathway, val_recall_superpathway, val_f1_superpathway, val_precision_class, val_recall_class, val_f1_class = evaluation_epoch(model, val_loader, criterion, device, experim_folder=EXPERIMENT_FOLDER)

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
                test_avg_loss, test_precision, test_recall, test_f1, test_precision_pathway, test_recall_pathway, test_f1_pathway, test_precision_superpathway, test_recall_superpathway, test_f1_superpathway, test_precision_class, test_recall_class, test_f1_class = evaluation_epoch(model, test_loader, criterion, device, mode="explain_test")

                final_stats(grid_statistics, config_idx, n_config, early_stopping.last_checkpoint_epoch)
                break

    return grid_statistics

def evaluate(config_path):
    # read json
    import json
    with open(config_path, 'r') as f:
        grid_config = json.load(f)

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

    for i in range(5):
        
        torch.manual_seed(i)
        train_loader, test_loader = prepare_dataloaders(MODEL_NAME, validation_set=False, seed=i)

        sample = next(iter(train_loader))
        num_node_features = sample.x.size(-1)
        edge_dim = sample.edge_attr.size(-1) if sample.edge_attr is not None else None
        num_classes = 730
        
        model = GIN(
            num_node_features=num_node_features,
            dim_h=grid_config['dim_h'],
            num_classes=num_classes,
            drop_rate=grid_config['drop_rate'],
            edge_dim=edge_dim,
            fingerprint=USE_FINGERPRINT,
            num_layers=4
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

        optimizer = optim.AdamW(model.parameters(), lr=grid_config['learning_rate'], weight_decay=grid_config['l2_rate'])
        criterion = nn.CrossEntropyLoss() if not USE_MULTILABEL else nn.BCEWithLogitsLoss()

        for epoch in range(40):
            print(f"[{MODEL_NAME.upper()}] Epoch {epoch+1}/{40}")
            start_time = time.time()
            train_loss, train_precision, train_recall, train_f1 = training_epoch(model, train_loader, optimizer, criterion, device, experim_folder=EXPERIMENT_FOLDER)
            end_time = time.time()

            grid_statistics['train_loss'].append(train_loss)
            grid_statistics['train_precision'].append(train_precision)
            grid_statistics['train_recall'].append(train_recall)
            grid_statistics['train_f1'].append(train_f1)
            grid_statistics['epoch_time'].append(end_time - start_time)
            
        test_avg_loss, test_precision, test_recall, test_f1, test_precision_pathway, test_recall_pathway, test_f1_pathway, test_precision_superpathway, test_recall_superpathway, test_f1_superpathway, test_precision_class, test_recall_class, test_f1_class = evaluation_epoch(model, test_loader, criterion, device, mode="explain_test")
        # add to a results file in a csv format using pandas
        import pandas as pd
        results_file = os.path.join(EXPERIMENT_FOLDER, "reports", f"evaluation_results.csv")
        if not os.path.exists(results_file):
            df = pd.DataFrame(columns=['run', 'test_loss', 'test_precision', 'test_recall', 'test_f1', 'test_precision_pathway', 'test_recall_pathway', 'test_f1_pathway', 'test_precision_superpathway', 'test_recall_superpathway', 'test_f1_superpathway', 'test_precision_class', 'test_recall_class', 'test_f1_class'])
            df.to_csv(results_file, index=False)
        df = pd.read_csv(results_file)
        df = pd.concat([df, pd.DataFrame({
            'run': [i],
            'test_loss': [test_avg_loss],
            'test_precision': [test_precision],
            'test_recall': [test_recall],
            'test_f1': [test_f1],
            'test_precision_pathway': [test_precision_pathway],
            'test_recall_pathway': [test_recall_pathway],
            'test_f1_pathway': [test_f1_pathway],
            'test_precision_superpathway': [test_precision_superpathway],
            'test_recall_superpathway': [test_recall_superpathway],
            'test_f1_superpathway': [test_f1_superpathway],
            'test_precision_class': [test_precision_class],
            'test_recall_class': [test_recall_class],
            'test_f1_class': [test_f1_class]
        })])
        df.to_csv(results_file, index=False)

    return grid_statistics




if __name__ == "__main__":
    # train_dataloader, val_dataloader, test_dataloader = prepare_dataloaders(MODEL_NAME)
    train_dataloader, test_dataloader = prepare_dataloaders(MODEL_NAME, validation_set=False)
    # train_dataloader = train_dataloader
    # val_dataloader = val_dataloader
    # test_dataloader = test_dataloader
    sample = next(iter(train_dataloader))
    num_node_features = sample.x.size(-1)
    edge_dim = sample.edge_attr.size(-1) if sample.edge_attr is not None else None
    num_classes = 730

    # Esempio di grid search manuale
    # from itertools import product
    # from config import PARAM_GRID_GRAPH
    # param_grid = PARAM_GRID_GRAPH
    # keys, values = zip(*param_grid.items())
    # configs = [dict(zip(keys, v)) for v in product(*values)]
    # n_config = len(configs)

    # for config_idx, grid_config in enumerate(configs, 1):
    #     print(f"\n=== CONFIG {config_idx}/{n_config} ===")
    #     run_experiment(grid_config, train_dataloader, val_dataloader, test_dataloader, num_node_features, edge_dim, num_classes, config_idx, n_config)
    config_path = "best_model_config_1_run_5_config.json"
    evaluate(config_path)


    