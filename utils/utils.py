import numpy as np
from config import DATADIR
from typing import Union, Literal
import pickle, shutil
from utils.fingerprint_handler import calculate_fingerprint

def data_generation(idx, data, 
                    spec_mol_smile=None, 
                    save_dataset:Union[Literal["train"], Literal["val"], Literal["test"]]=None):
    # Generate docstring
    """
    Generate data for training, validation, and test set
    Parameters
    ----------
    idx : list
        List of indices of the molecules to be selected
    data : dict
        Dictionary of the dataset
    spec_mol_smile : str, optional
        Specific molecule to be excluded from the dataset, by default None
    save_dataset : Union[Literal["train"], Literal["val"], Literal["test"]], optional
        Save the dataset to a file, by default None
    Returns
    -------
    list
        List of SMILES
    list
        List of fingerprints
    list
        List of pathways
    list
        List of superclass
    list
        List of class
    """

    smiles_list = []
    Y_train_path = []
    Y_train_super = []
    Y_train_class = []
    m_fingerprint_list = []

    for i, n in enumerate(idx):
        smiles = data[n]['SMILES']
        # if MODEL.lower() == "mlp":
        m_fingerprint_list.append(np.concatenate(calculate_fingerprint(smiles, 2), axis=1))
        if spec_mol_smile is not None:
            if smiles != spec_mol_smile:
                pass
        smiles_list.append(smiles)
        Y_train_path.append(data[n]['Pathway'])
        Y_train_super.append(data[n]['Super_class'])
        Y_train_class.append(data[n]['Class'])

    if save_dataset:
        dictionary = {'smiles': smiles_list,
                    'fingerprint': m_fingerprint_list,
                    'y_path': Y_train_path,
                    'y_super': Y_train_super,
                    'y_class': Y_train_class} 
        with open(f'{DATADIR}/{save_dataset}_dataset.pkl', 'wb') as f:
            pickle.dump(dictionary, f)
        print(f"{save_dataset} dataset saved successfully.")
    return smiles_list, m_fingerprint_list, Y_train_path, Y_train_super, Y_train_class 


def select_class_idx_path(target_vectors: list, 
                     smiles: list, 
                     m_fingrprints: list=None,
                     class_indexes: list=None, 
                     label_enc:Union[Literal["onehot"], Literal["one-hot"]]="onehot",
                     n_samples:int=None, 
                     shuffle:bool=True
                     ):
    
    from config import LABELS_CODES
    datasplit_targets = []
    datasplit_smiles_classes = []
    datasplit_m_fingerprints = []
    #TODO: sistemare dopo che decidiamo se aumentare il numero di classi o no
    #num_classes = len(target_vectors[0])
    num_classes = len(class_indexes) if class_indexes is not None else len(LABELS_CODES.keys())
    
    for label_key, label_val in LABELS_CODES.items(): 
        # idx_tmp = np.argwhere(np.array(target_vectors)==label_val)[:, 0]
        idx_tmp = np.argwhere(np.array([str(v) for v in np.array(target_vectors)])==np.array([str(vv) for vv in label_val[None,:]]))  
        if n_samples is not None and n_samples <= len(idx_tmp):
            idx_ = idx_tmp[:n_samples]  # idx_.shape = (batch_size, 1)
            print(f"N_SAMPLES available for class {label_key}, {n_samples} examples taken.")
        else:
            print(f"N_SAMPLES not available for class {label_key}, {len(idx_tmp)} examples taken.")
            n_samples = len(idx_tmp)
            idx_ = idx_tmp
        
        idx_ = idx_.flatten()       # idx_.shape = (batch_size,)
        
        # datasplit_targets.append(np.array(target_vectors)[idx_])
        if "hot" in label_enc:
            datasplit_targets.append(np.array([np.eye(num_classes)[label_key]]*n_samples))
        else:
            datasplit_targets.append([label_key]*n_samples)

        datasplit_smiles_classes.append(np.array(smiles)[idx_])

        datasplit_m_fingerprints.append(np.array(m_fingrprints)[idx_])
        # if MODEL.lower() == "mlp" and m_fingrprints is not None:
        #     datasplit_m_fingerprints.append(np.array(m_fingrprints)[idx_])
        # else:
        #     datasplit_m_fingerprints = None
    
    # Randomize the order of the samples
    datasplit_targets = np.concatenate(datasplit_targets)
    datasplit_smiles_classes = np.concatenate(datasplit_smiles_classes)
    # if datasplit_m_fingerprints is not None and len(datasplit_m_fingerprints) > 0:
    #     datasplit_m_fingerprints = np.concatenate(datasplit_m_fingerprints)
    datasplit_m_fingerprints = np.concatenate(datasplit_m_fingerprints)
    indexes = np.arange(len(datasplit_smiles_classes)).astype(int)
    if shuffle:
        np.random.shuffle(indexes)
    if datasplit_m_fingerprints is not None:
        return datasplit_targets[indexes], datasplit_smiles_classes[indexes], datasplit_m_fingerprints[indexes]
    else:
        return datasplit_targets[indexes], datasplit_smiles_classes[indexes], None


# def select_class_idx(target_vectors: list, 
#                      smiles: list, 
#                      m_fingrprints: list=None,
#                      class_indexes: list=None, 
#                      label_enc:Union[Literal["onehot"], Literal["one-hot"]]="onehot",
#                      n_samples:int=None, 
#                      shuffle:bool=True
#                      ):
    
#     datasplit_targets = []
#     datasplit_smiles_classes = []
#     datasplit_m_fingerprints = []
#     #TODO: sistemare dopo che decidiamo se aumentare il numero di classi o no
#     num_classes = len(target_vectors[0])
#     #select indices of the single class in target vector

#     idx_single_class = np.argwhere(np.array(target_vectors)[np.argwhere(np.sum(target_vectors, axis=1)==1)[:,0]])[:,1]
#     idx_tmp = idx_single_class


#     if n_samples is not None and n_samples <= len(idx_tmp):
#         idx_ = idx_tmp[:n_samples]  # idx_.shape = (batch_size, 1)
#         print(f"N_SAMPLES available for class {idx_tmp}, {n_samples} examples taken.")
#     else:
#         print(f"N_SAMPLES not available for class {idx_tmp}, {len(idx_)} examples taken.")
#         n_samples = len(idx_)
    
#     datasplit_smiles_classes = np.array(smiles)[idx_single_class].tolist()

    


def select_class_idx_super(target_vectors: list, 
                     smiles: list, 
                     m_fingrprints: list=None,
                     class_indexes: list=None, 
                     label_enc:Union[Literal["onehot"], Literal["one-hot"]]="onehot",
                     n_samples:int=None, 
                     shuffle:bool=True
                    ):
    
    num_classes = len(target_vectors[0])
    # save Single class idx
    idx_single_class = np.argwhere(np.sum(target_vectors, axis=1)==1)
    
    # transformation in numpy array
    smiles, target_vectors = np.array(smiles), np.array(target_vectors)

    # save only values appartaining to Single calss 
    target_vectors = target_vectors[idx_single_class]
    smiles = smiles[idx_single_class]

    # apply argmax to recover the index (not one-hot encoded) target for the samples
    datasplit_path_single_class = np.argmax(target_vectors, axis=2)

    datasplit_targets = []
    datasplit_smiles_classes = []
    datasplit_m_fingerprints = []
    
    printed = False
    # Find unique values in target arrays
    unique_labels = np.unique(datasplit_path_single_class)
    # Iterate over unique labels and select samples
    for i, label in enumerate(unique_labels): 
        if printed == False:
            print(f"Class {i}: {label}")
            printed = True

        idx_tmp = np.argwhere(datasplit_path_single_class==label)[:, 0]  
        
        if n_samples is not None and n_samples <= len(idx_tmp):
            idx_ = idx_tmp[:n_samples]
        else:
            print(f"N_SAMPLES not available for class {label}, {len(idx_tmp)} examples taken.")
            idx_ = idx_tmp
        
        # datasplit_targets.append(datasplit_path_single_class[idx_])
        # datasplit_smiles_classes.append(smiles[idx_])
       
        idx_ = idx_.flatten() 

        datasplit_smiles_classes.append(np.array(smiles)[idx_])

        datasplit_m_fingerprints.append(np.array(m_fingrprints)[idx_])

        datasplit_targets.append(np.array(target_vectors)[idx_])
        #  labels = np.zeros((datasplit_targets.shape[0], label_translated.shape[0]))
        #  labels[np.argwhere(datasplit_targets == label)[:, 0]] = i

        printed = False
    # idx_shik = np.argwhere(datasplit_path_single_class==5)[:, 0]   #Class: Shikimates
    # datasplit_path_binary_shik = datasplit_path_single_class[idx_shik]
    # datasplit_smiles_binary_shik = smiles[idx_shik]
    datasplit_targets = np.concatenate(datasplit_targets, axis=0)
    datasplit_smiles_classes = np.concatenate(datasplit_smiles_classes, axis=0)
    datasplit_m_fingerprints = np.concatenate(datasplit_m_fingerprints, axis=0)
    # for i, label in enumerate(class_number):
    #     if label_type == "two_classes": #and len(class_number) == 2:
    #         datasplit_targets[np.argwhere(datasplit_targets == label)[:, 0]] = i    # 0 or 1
    #         paths_classes_to_return = datasplit_targets[:, 0]

    #     elif label_type == "ohe" or label_type == "binary":
    #         if i == 0:
    #             labels = np.zeros((len(datasplit_targets), num_classes))
    #         labels[np.argwhere(datasplit_targets == label)[:, 0], :] = np.eye(num_classes)[i]
    #         paths_classes_to_return = labels

    indexes = np.arange(len(datasplit_smiles_classes)).astype(int)
    if shuffle:
        np.random.shuffle(indexes)
    if datasplit_m_fingerprints is not None:
        return datasplit_targets[indexes], datasplit_smiles_classes[indexes], datasplit_m_fingerprints[indexes]
    else:
        return datasplit_targets[indexes], datasplit_smiles_classes[indexes], None
    
    
def multilabel_classification_report(y_true, y_pred, target_names=None, 
                                   digits=4, zero_division='warn'):
    """
    Genera un report di classificazione per problemi multilabel.
    
    Parameters:
    -----------
    y_true : array-like of shape (n_samples, n_labels)
        Etichette vere (0 o 1)
    y_pred : array-like of shape (n_samples, n_labels)  
        Etichette predette (0 o 1)
    target_names : list, optional
        Nomi delle classi/etichette
    digits : int, default=2
        Numero di cifre decimali
    zero_division : str, default='warn'
        Come gestire la divisione per zero
    
    Returns:
    --------
    str : Report formattato
    """
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    n_samples, n_labels = y_true.shape
    
    if target_names is None:
        target_names = [f'label_{i}' for i in range(n_labels)]
    
    # Calcola metriche per ogni label
    metrics = {}
    
    for i in range(n_labels):
        true_label = y_true[:, i]
        pred_label = y_pred[:, i]
        
        # True/False Positives/Negatives
        tp = np.sum((true_label == 1) & (pred_label == 1))
        fp = np.sum((true_label == 0) & (pred_label == 1))
        fn = np.sum((true_label == 1) & (pred_label == 0))
        tn = np.sum((true_label == 0) & (pred_label == 0))
        
        # Precision, Recall, F1-score
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        support = np.sum(true_label == 1)
        
        metrics[target_names[i]] = {
            'precision': precision,
            'recall': recall,
            'f1-score': f1,
            'support': support
        }
    
    # Calcola metriche aggregate
    # Micro average
    total_tp = sum(np.sum((y_true[:, i] == 1) & (y_pred[:, i] == 1)) for i in range(n_labels))
    total_fp = sum(np.sum((y_true[:, i] == 0) & (y_pred[:, i] == 1)) for i in range(n_labels))
    total_fn = sum(np.sum((y_true[:, i] == 1) & (y_pred[:, i] == 0)) for i in range(n_labels))
    
    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    micro_f1 = 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0
    
    # Macro average
    macro_precision = np.mean([metrics[label]['precision'] for label in target_names])
    macro_recall = np.mean([metrics[label]['recall'] for label in target_names])
    macro_f1 = np.mean([metrics[label]['f1-score'] for label in target_names])
    
    # Weighted average
    total_support = sum(metrics[label]['support'] for label in target_names)
    if total_support > 0:
        weighted_precision = sum(metrics[label]['precision'] * metrics[label]['support'] 
                               for label in target_names) / total_support
        weighted_recall = sum(metrics[label]['recall'] * metrics[label]['support'] 
                            for label in target_names) / total_support
        weighted_f1 = sum(metrics[label]['f1-score'] * metrics[label]['support'] 
                        for label in target_names) / total_support
    else:
        weighted_precision = weighted_recall = weighted_f1 = 0
    
    # Formatta il report
    headers = ['precision', 'recall', 'f1-score', 'support']
    name_width = max(len(name) for name in target_names + ['micro avg', 'macro avg', 'weighted avg'])
    width = max(name_width, digits + 5)
    
    # Header
    report = f"{'':>{name_width}} "
    for header in headers:
        report += f"{header:>{width}}"
    report += "\n\n"
    
    # Per-class metrics
    for label in target_names:
        report += f"{label:>{name_width}} "
        report += f"{metrics[label]['precision']:{width}.{digits}f}"
        report += f"{metrics[label]['recall']:{width}.{digits}f}"
        report += f"{metrics[label]['f1-score']:{width}.{digits}f}"
        report += f"{metrics[label]['support']:{width}}"
        report += "\n"
    
    report += "\n"
    
    # Aggregate metrics
    aggregates = [
        ('micro avg', micro_precision, micro_recall, micro_f1, total_support),
        ('macro avg', macro_precision, macro_recall, macro_f1, total_support),
        ('weighted avg', weighted_precision, weighted_recall, weighted_f1, total_support)
    ]
    
    for name, prec, rec, f1, supp in aggregates:
        report += f"{name:>{name_width}} "
        report += f"{prec:{width}.{digits}f}"
        report += f"{rec:{width}.{digits}f}"
        report += f"{f1:{width}.{digits}f}"
        report += f"{supp:{width}}"
        report += "\n"
    
    return report


from typing import Literal
import os
import datetime

def initialize_experiment(
    models: Literal["mlp", "gin", "gine", "gat", "gate"],
    target_type: Literal["pathway", "superclass", "class"],
    base_dir: str = os.path.dirname(os.path.abspath(__file__))
    ):
    
    MODELS = models
    TARGET_TYPE = target_type
    BASEDIR = base_dir
        
    # Initialize experiment folder 
    EXPERIMENT_FOLDER = os.path.join(BASEDIR, "experiments", MODELS + "_" + TARGET_TYPE.lower() + "_" +datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(EXPERIMENT_FOLDER, exist_ok=True)

    # Create a models folder
    os.makedirs(os.path.join(EXPERIMENT_FOLDER, "models"), exist_ok=True)
    os.makedirs(os.path.join(EXPERIMENT_FOLDER, "utils"), exist_ok=True)
    # Create the weights folder
    os.makedirs(os.path.join(EXPERIMENT_FOLDER, "pt"), exist_ok=True)
    # Create the reports folder
    os.makedirs(os.path.join(EXPERIMENT_FOLDER, "reports"), exist_ok=True)

    # Save the configuration file
    # shutil.copy(__file__, os.path.join(EXPERIMENT_FOLDER, "config_gridsearch.py"))

    # Save the models file from the directory
    for model_file in os.listdir(os.path.join(BASEDIR, "models")):
        if model_file.endswith(".py"):
            shutil.copy(os.path.join(BASEDIR, "models", model_file), os.path.join(EXPERIMENT_FOLDER, "models", model_file))
            
    # Copy all the py files in the utils folder
    for file in os.listdir(os.path.join(BASEDIR, "utils")):
        if file.endswith(".py"):
            shutil.copy(os.path.join(BASEDIR, "utils", file), os.path.join(EXPERIMENT_FOLDER, "utils", file))
            
    # Copy all the py files in the folder 
    for file in os.listdir(BASEDIR):
        if file.endswith(".py"):
            shutil.copy(os.path.join(BASEDIR, file), os.path.join(EXPERIMENT_FOLDER, file))
            
    return EXPERIMENT_FOLDER


import torch

def final_stats(statistics: dict, config_idx: int, n_config: int, last_checkpoint_epoch: int = None):
    """
    Print the final statistics of the training and validation process.
    
    Args:
        statistics (dict): A dictionary containing the training and validation statistics.
        config_idx (int): The index of the current configuration.
        n_config (int): The total number of configurations.
    """
    
    if last_checkpoint_epoch is None:
        last_checkpoint_epoch = len(statistics['train_loss'])
    
    avg_train_loss = torch.mean(torch.tensor(statistics['train_loss'][:last_checkpoint_epoch]))
    avg_train_precision = torch.mean(torch.tensor(statistics['train_precision'][:last_checkpoint_epoch]))
    avg_train_recall = torch.mean(torch.tensor(statistics['train_recall'][:last_checkpoint_epoch]))
    avg_train_f1 = torch.mean(torch.tensor(statistics['train_f1'][:last_checkpoint_epoch]))
    std_train_loss = torch.std(torch.tensor(statistics['train_loss'][:last_checkpoint_epoch]))
    std_train_precision = torch.std(torch.tensor(statistics['train_precision'][:last_checkpoint_epoch]))
    std_train_recall = torch.std(torch.tensor(statistics['train_recall'][:last_checkpoint_epoch]))
    std_train_f1 = torch.std(torch.tensor(statistics['train_f1'][:last_checkpoint_epoch]))
    avg_val_loss = torch.mean(torch.tensor(statistics['val_loss'][:last_checkpoint_epoch]))
    avg_val_precision = torch.mean(torch.tensor(statistics['val_precision'][:last_checkpoint_epoch]))
    avg_val_recall = torch.mean(torch.tensor(statistics['val_recall'][:last_checkpoint_epoch]))
    avg_val_f1 = torch.mean(torch.tensor(statistics['val_f1'][:last_checkpoint_epoch]))
    std_val_loss = torch.std(torch.tensor(statistics['val_loss'][:last_checkpoint_epoch]))
    std_val_precision = torch.std(torch.tensor(statistics['val_precision'][:last_checkpoint_epoch]))
    std_val_recall = torch.std(torch.tensor(statistics['val_recall'][:last_checkpoint_epoch]))
    std_val_f1 = torch.std(torch.tensor(statistics['val_f1'][:last_checkpoint_epoch]))
    avg_epoch_time = torch.mean(torch.tensor(statistics['epoch_time'][:last_checkpoint_epoch]))

    final_log_train = f"[CONFIG {config_idx}/{n_config}] Train Loss: {avg_train_loss:.4f} ± {std_train_loss:.4f}, Precision: {avg_train_precision:.4f} ± {std_train_precision:.4f}, Recall: {avg_train_recall:.4f} ± {std_train_recall:.4f}, F1: {avg_train_f1:.4f} ± {std_train_f1:.4f}, Epoch Time: {avg_epoch_time:.2f} seconds"
    
    final_log_val = f"[CONFIG {config_idx}/{n_config}] Val Loss: {avg_val_loss:.4f} ± {std_val_loss:.4f}, Precision: {avg_val_precision:.4f} ± {std_val_precision:.4f}, Recall: {avg_val_recall:.4f} ± {std_val_recall:.4f}, F1: {avg_val_f1:.4f} ± {std_val_f1:.4f}"

    print("Final Training Summary:", final_log_train)
    print("Final Validation Summary:", final_log_val)