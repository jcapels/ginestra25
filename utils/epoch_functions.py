import torch, numpy as np
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score, f1_score
from utils.topk import top_k_accuracy, top_k_coverage
from config import USE_FINGERPRINT, ATOMS_LIST as permitted_atoms, TARGETS_LIST
from utils.explainability import explain_all_molecules_in_batches
from typing import Union, Literal
from utils.utils import multilabel_classification_report
import os

def training_epoch(model, dataloader, optimizer, criterion, device):
    """
    Training loop for the model.
    Args:
    model: the model to train
    dataloader: the training dataloader
    optimizer: the optimizer to use
    criterion: the loss function
    device: the device to use (cpu or cuda)
    cumulative_loss: if True, the loss will be the sum of the CrossEntropyLoss and the cosine similarity loss

    Returns:
    avg_loss: the average loss over the training set
    loss: the loss of the last batch

    """
    model.train()
    total_loss = 0.0
    all_preds, all_targets = [], []

    for b, batch in enumerate(dataloader):
        if "MLP" in model.__class__.__name__:
            # Fingerprint at index 1 (shape [batch, 1, feature_dim]); target at index 2
            x, y = batch[1].to(device).float(), batch[2].to(device).float()
            x = x.squeeze(1) if x.dim() == 3 else x  # [batch, feature_dim]
            optimizer.zero_grad()
            out = model(x)
        elif model.__class__.__name__ == "GIN" or model.__class__.__name__ == "GAT" or model.__class__.__name__ == "GCN":
            x, y, edge_index, batch_ = batch.x.to(device).float(), batch.y.to(device).float(), batch.edge_index.to(device), batch.batch.to(device)
            optimizer.zero_grad()
            out = model(x, edge_index=edge_index, batch=batch_, fingerprint=batch.fingerprint_tensor.to(device) if USE_FINGERPRINT else None) # [batch, num_classes]
        elif model.__class__.__name__ == "GINE" or model.__class__.__name__ == "GATE":
            x, y, edge_index, edge_attr, batch_ = batch.x.to(device).float(), batch.y.to(device).float(), batch.edge_index.to(device), batch.edge_attr.to(device), batch.batch.to(device)
            optimizer.zero_grad()
            out = model(x, edge_index=edge_index, edge_attr=edge_attr, batch=batch_, fingerprint=batch.fingerprint_tensor.to(device) if USE_FINGERPRINT else None)
        if isinstance(criterion, torch.nn.CrossEntropyLoss):
            loss = criterion(out, y.argmax(dim=1))
        elif isinstance(criterion, torch.nn.BCEWithLogitsLoss) or isinstance(criterion, torch.nn.MultiLabelSoftMarginLoss):
            loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        all_preds.extend(out.detach().cpu().numpy())
        all_targets.extend(y.detach().cpu().numpy())
    print("===============================================================================")
    print(f"TRAINING REPORT")
    print("===============================================================================")
    # Compute the classification report
    print(multilabel_classification_report(
        y_true=np.array(all_targets),
        y_pred=np.array((F.sigmoid(torch.Tensor(np.array(all_preds))) > 0.5)),
        target_names=TARGETS_LIST,
    ))
    avg_loss = total_loss / len(dataloader)
    precision = precision_score(all_targets, (F.sigmoid(torch.Tensor(np.array(all_preds)))>0.5), average='macro', zero_division=0)
    recall = recall_score(all_targets, (F.sigmoid(torch.Tensor(np.array(all_preds)))>0.5), average='macro', zero_division=0)
    f1 = f1_score(all_targets, (F.sigmoid(torch.Tensor(np.array(all_preds)))>0.5), average='macro', zero_division=0)

    return avg_loss, precision, recall, f1


def evaluation_epoch(model, dataloader, criterion, device, mode:Union[Literal["explain_val", "explain_test"], None]=None, **kwargs):
    """
    Evaluation loop for the model.
    Args:
    model: the model to evaluate
    dataloader: the evaluation dataloader
    device: the device to use (cpu or cuda)
    verbose: if True, print the classification report

    Returns:
    avg_loss: the average loss over the evaluation set
    loss: the loss of the last batch
    """

        
    model.eval()
    if mode is not None and "explain" in mode and "val" in mode:
        suffix = "val"
    elif mode is not None and "explain" in mode and "test" in mode:
        suffix = "test"
    else:
        suffix = None    
        
    if "experim_folder" in kwargs:
        experim_folder = kwargs["experim_folder"]
        output_dir = os.path.join(experim_folder, "explainability", f"{suffix}_explanations")
        os.makedirs(output_dir, exist_ok=True)
    else:
        experim_folder = os.path.join(os.getcwd(), "explainability")
        output_dir = os.path.join(experim_folder, f"{suffix}_explanations")
        os.makedirs(output_dir, exist_ok=True)
        
    if suffix is not None:
        from config import DATADIR, TARGET_TYPE
        # Esegui analisi completa
        explain_all_molecules_in_batches(
            model=model,
            test_dataloader=dataloader,
            device=device,
            permitted_atoms=permitted_atoms,
            max_molecules=10,
            smiles_file=f'{DATADIR}/{suffix}_smiles_{TARGET_TYPE}.csv',
            output_dir=output_dir,
        )
    total_loss = 0.0
    all_preds, all_targets = [], []
    topk = {}

    with torch.no_grad():
        for b, batch in enumerate(dataloader):
            if "MLP" in model.__class__.__name__:
                # Fingerprint at index 1; target at index 2
                x, y = batch[1].to(device).float(), batch[2].to(device).float()
                x = x.squeeze(1) if x.dim() == 3 else x  # [batch, feature_dim]
                out = model(x)
            elif model.__class__.__name__ == "GIN" or model.__class__.__name__ == "GAT" or model.__class__.__name__ == "GCN":
                x, y, edge_index, batch_ = batch.x.to(device).float(), batch.y.to(device).float(), batch.edge_index.to(device), batch.batch.to(device)
                out = model(x, edge_index=edge_index, batch=batch_, fingerprint=batch.fingerprint_tensor.to(device) if USE_FINGERPRINT else None) # [batch, num_classes]
            elif model.__class__.__name__ == "GINE" or model.__class__.__name__ == "GATE":
                x, y, edge_index, edge_attr, batch_ = batch.x.to(device).float(), batch.y.to(device).float(), batch.edge_index.to(device), batch.edge_attr.to(device), batch.batch.to(device)
                out = model(x, edge_index=edge_index, edge_attr=edge_attr, batch=batch_, fingerprint=batch.fingerprint_tensor.to(device) if USE_FINGERPRINT else None)
            if isinstance(criterion, torch.nn.CrossEntropyLoss):
                loss = criterion(out, y.argmax(dim=1))
            elif isinstance(criterion, torch.nn.BCEWithLogitsLoss) or isinstance(criterion, torch.nn.MultiLabelSoftMarginLoss):
                loss = criterion(out, y)
            total_loss += loss.item()
            all_preds.extend(out.cpu().numpy())
            all_targets.extend(y.cpu().numpy())

        print("===============================================================================")
        print(f"EVALUATION REPORT")
        print("===============================================================================")
        # Compute the classification report
        print(multilabel_classification_report(
            y_true=np.array(all_targets),
            y_pred=np.array((F.sigmoid(torch.Tensor(np.array(all_preds))) > 0.5)),
            target_names=TARGETS_LIST,
        ))
        avg_loss = total_loss / len(dataloader)
        precision = precision_score(all_targets, (F.sigmoid(torch.Tensor(np.array(all_preds))) > 0.5), average='macro', zero_division=0)
        recall = recall_score(all_targets, (F.sigmoid(torch.Tensor(np.array(all_preds))) > 0.5), average='macro', zero_division=0)
        f1 = f1_score(all_targets, (F.sigmoid(torch.Tensor(np.array(all_preds))) > 0.5), average='macro', zero_division=0)
        topk['1'] = top_k_accuracy(out, y, k=1)
        topk['3'] = top_k_accuracy(out, y, k=3)
        topk['5'] = top_k_accuracy(out, y, k=5)
        topk['1_coverage'] = top_k_coverage(out, y, k=1)
        topk['3_coverage'] = top_k_coverage(out, y, k=3)
        topk['5_coverage'] = top_k_coverage(out, y, k=5)

    return avg_loss, precision, recall, f1, topk
