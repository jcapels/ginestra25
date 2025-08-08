import torch
import numpy as np
import argparse
import json
import os
from models.GCN import GCN
from models.GAT import GAT
from models.GIN import GIN
from models.GINE import GINE
from models.GATE import GATE
from models.MLP import MLP
from config import DEVICE, LABELS_CODES, USE_FINGERPRINT, TARGETS_LIST

from rdkit import Chem
from torch_geometric.data import Data
from utils.graph_data_def import get_atom_features, get_bond_features

def preprocess_single_molecule(smiles_string):
    """
    Converts a SMILES string into a torch_geometric.data.Data object,
    using the same featurization as in training.
    """
    mol = Chem.MolFromSmiles(smiles_string)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles_string}")

    n_nodes = mol.GetNumAtoms()

    # Get feature dimensions from a sample atom/bond
    n_node_features = len(get_atom_features(mol.GetAtomWithIdx(0)))
    if mol.GetNumBonds() > 0:
        n_edge_features = len(get_bond_features(mol.GetBondWithIdx(0)))
    else:
        # For molecules with no bonds, use a default edge feature size
        n_edge_features = 6  # Adjust based on your bond feature implementation

    X = []
    for atom in mol.GetAtoms():
        X.append(get_atom_features(atom))
    X = np.array(X)  # Convert list of arrays to a single numpy array
    X = torch.tensor(X, dtype=torch.float)

    # Edge index and edge features
    adj_matrix = Chem.rdmolops.GetAdjacencyMatrix(mol)
    rows, cols = adj_matrix.nonzero()
    edge_index = []
    edge_attr = []

    for i, j in zip(rows, cols):
        bond = mol.GetBondBetweenAtoms(int(i), int(j))
        if bond is not None:
            edge_index.append([i, j])
            edge_attr.append(get_bond_features(bond))

    if edge_index:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, n_edge_features), dtype=torch.float)

    # Batch (single graph)
    batch = torch.zeros(X.size(0), dtype=torch.long)

    # Create data object
    data_args = {
        "x": X,
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "batch": batch
    }

    # Optionally add fingerprint (set to zeros if not used during inference)
    if USE_FINGERPRINT:
        # You may want to compute the actual fingerprint here
        # For now, we use zeros as placeholder
        data_args["fingerprint_tensor"] = torch.zeros(2048)  # Adjust size as needed

    return Data(**data_args).to(DEVICE)


def load_model_config(model_path):
    """
    Load model configuration from the corresponding _config.json file.

    Args:
        model_path (str): Path to the model .pt file

    Returns:
        dict: Configuration dictionary
    """
    # Generate config file path by replacing .pt with _config.json
    config_path = model_path.replace('.pt', '_config.json')

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        print(f"✅ Loaded configuration from: {config_path}")
        return config
    except Exception as e:
        raise RuntimeError(f"Failed to load configuration: {e}")


def get_smiles_input():
    """
    Prompts the user to input a SMILES string with validation.
    """
    while True:
        smiles = input("\nEnter the SMILES string of the molecule to classify: ").strip()

        if not smiles:
            print("❌ Empty SMILES. Please try again.")
            continue

        # Basic validation with RDKit
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                print(f"❌ Invalid SMILES: '{smiles}'. Please try again.")
                continue
            else:
                print(f"✅ Valid SMILES: '{smiles}'")
                return smiles
        except Exception as e:
            print(f"❌ Error validating SMILES: {e}. Please try again.")
            continue


def main(args):
    # Get SMILES input
    if args.input_smiles:
        input_smiles = args.input_smiles
        print(f"Using SMILES from argument: {input_smiles}")
    else:
        print("="*60)
        print("🧪 GINESTRA - Molecular Classification Inference")
        print("="*60)
        input_smiles = get_smiles_input()

    # Load model configuration
    try:
        config = load_model_config(args.model_path.replace('.pt', '_config.json'))
    except Exception as e:
        print(f"❌ {e}")
        print("💡 Make sure the _config.json file exists alongside your model file.")
        return

    # Get sample data to determine feature dimensions
    sample_mol = Chem.MolFromSmiles("CCO")  # Simple ethanol molecule
    num_node_features = len(get_atom_features(sample_mol.GetAtomWithIdx(0)))
    edge_dim = len(get_bond_features(sample_mol.GetBondWithIdx(0))) if sample_mol.GetNumBonds() > 0 else 6
    num_classes = len(LABELS_CODES)
    
    if "gcn" in args.model_type:
        # Extract model parameters from config
        dim_h = config.get("dim_h", 64)
        drop_rate = config.get("drop_rate", 0.5)
        num_layers = config.get("num_layers", 2)
        print(f"📋 Model Configuration:")
        print(f"   - Hidden dimension: {dim_h}")
        print(f"   - Dropout rate: {drop_rate}")
        print(f"   - Number of layers: {num_layers}")
        print(f"   - Node features: {num_node_features}")
        print(f"   - Edge features: {edge_dim}")
        print(f"   - Number of classes: {num_classes}")
        model_map = {
            "gcn": GCN(
            num_node_features=num_node_features, 
            dim_h=dim_h, 
            num_classes=num_classes,
            drop_rate=drop_rate,
            fingerprint=USE_FINGERPRINT,
            num_layers=num_layers
            )
        }
    elif "gat" in args.model_type:
        # Extract model parameters from config
        dim_h = config.get("dim_h", 64)
        drop_rate = config.get("drop_rate", 0.5)
        num_layers = config.get("num_layers", 2)
        edge_dim = config.get("edge_dim", 6)
        print(f"📋 Model Configuration:")
        print(f"   - Hidden dimension: {dim_h}")
        print(f"   - Dropout rate: {drop_rate}")
        print(f"   - Number of layers: {num_layers}")
        print(f"   - Node features: {num_node_features}")
        print(f"   - Edge features: {edge_dim}")
        print(f"   - Number of classes: {num_classes}")
        model_map = {
            "gat": GAT(
            num_node_features=num_node_features, 
            dim_h=dim_h, 
            num_classes=num_classes,
            drop_rate=drop_rate,
            fingerprint=USE_FINGERPRINT,
            num_layers=num_layers
            )
        }
    elif "gin" in args.model_type:
        # Extract model parameters from config
        dim_h = config.get("dim_h", 64)
        drop_rate = config.get("drop_rate", 0.5)
        num_layers = config.get("num_layers", 2)
        edge_dim = config.get("edge_dim", 6)
        print(f"📋 Model Configuration:")
        print(f"   - Hidden dimension: {dim_h}")
        print(f"   - Dropout rate: {drop_rate}")
        print(f"   - Number of layers: {num_layers}")
        print(f"   - Node features: {num_node_features}")
        print(f"   - Edge features: {edge_dim}")
        print(f"   - Number of classes: {num_classes}")
         # Initialize the model based on the type
        model_map = {
            "gin": GIN(
            num_node_features=num_node_features, 
            dim_h=dim_h, 
            num_classes=num_classes, 
            edge_dim=edge_dim,
            drop_rate=drop_rate,
            fingerprint=USE_FINGERPRINT,
            num_layers=num_layers
            )
        }
    elif "gine" in args.model_type:
        # Extract model parameters from config
        dim_h = config.get("dim_h", 64)
        drop_rate = config.get("drop_rate", 0.5)
        num_layers = config.get("num_layers", 2)
        edge_dim = config.get("edge_dim", 6)
        print(f"📋 Model Configuration:")
        print(f"   - Hidden dimension: {dim_h}")
        print(f"   - Dropout rate: {drop_rate}")
        print(f"   - Number of layers: {num_layers}")
        print(f"   - Node features: {num_node_features}")
        print(f"   - Edge features: {edge_dim}")
        print(f"   - Number of classes: {num_classes}")
        # Initialize the model based on the type
        model_map = {
            "gine": GINE(
            num_node_features=num_node_features, 
            dim_h=dim_h, 
            num_classes=num_classes, 
            edge_dim=edge_dim,
            drop_rate=drop_rate,
            fingerprint=USE_FINGERPRINT,
            num_layers=num_layers
            )
        }
    elif "gate" in args.model_type:
        # Extract model parameters from config
        dim_h = config.get("dim_h", 64)
        drop_rate = config.get("drop_rate", 0.5)
        num_layers = config.get("num_layers", 2)
        edge_dim = config.get("edge_dim", 6)
        print(f"📋 Model Configuration:")
        print(f"   - Hidden dimension: {dim_h}")
        print(f"   - Dropout rate: {drop_rate}")
        print(f"   - Number of layers: {num_layers}")
        print(f"   - Node features: {num_node_features}")
        print(f"   - Edge features: {edge_dim}")
        print(f"   - Number of classes: {num_classes}")
        # Initialize the model based on the type
        model_map = {
            "gate": GATE(
            num_node_features=num_node_features, 
            dim_h=dim_h, 
            num_classes=num_classes, 
            edge_dim=edge_dim,
            drop_rate=drop_rate,
            fingerprint=USE_FINGERPRINT,
            num_layers=num_layers
            )
        }
    elif "mlp" in args.model_type:
        # Extract model parameters from config
        unit1 = config.get("unit1", 128)
        unit2 = config.get("unit2", 64)
        unit3 = config.get("unit3", 32)
        drop_rate = config.get("drop_rate", 0.5)
        print(f"📋 Model Configuration:")
        print(f"   - Hidden dimension: {dim_h}")
        print(f"   - Dropout rate: {drop_rate}")
        print(f"   - Number of layers: {num_layers}")
        print(f"   - Node features: {num_node_features}")
        print(f"   - Edge features: {edge_dim}")
        print(f"   - Number of classes: {num_classes}")
        # Initialize the model based on the type
        model_map = {
            "mlp": MLP(
                unit1=unit1,
                unit2=unit2,
                unit3=unit3,
                drop_rate=drop_rate,
                num_classes=num_classes
            )
        }

    if args.model_type not in model_map:
        raise ValueError(f"Model type '{args.model_type}' is not supported.")

    model = model_map[args.model_type].to(DEVICE)
    print(f"\n🔧 Loaded architecture: {args.model_type.upper()}")

    # Load the trained model weights
    try:
        model.load_state_dict(torch.load(args.model_path, map_location=DEVICE))
        model.eval()  # Set model to evaluation mode
        print(f"✅ Loaded weights from: {args.model_path}")
    except Exception as e:
        raise RuntimeError(f"❌ Failed to load model weights: {e}")

    # Prepare the input data
    try:
        input_data = preprocess_single_molecule(input_smiles)
        print(f"✅ Successfully preprocessed molecule")
    except Exception as e:
        raise RuntimeError(f"❌ Failed to preprocess molecule: {e}")

    # Run inference
    print(f"\n🚀 Running inference...")
    with torch.no_grad():
        try:
            # Forward pass
            if args.model_type == "mlp":
                # MLP doesn't use edge information
                output = model(input_data.x, input_data.batch)
            elif args.model_type == "gine":
                # GINE requires edge attributes
                output = model(input_data.x, input_data.edge_index, input_data.edge_attr, input_data.batch)
            elif args.model_type == "gate":
                # GATE requires edge attributes
                output = model(input_data.x, input_data.edge_index, input_data.edge_attr, input_data.batch)
            elif args.model_type == "gcn":
                # GCN requires edge index and node features
                output = model(input_data.x, input_data.edge_index, input_data.batch)
            elif args.model_type == "gat":
                # GAT requires edge index and node features
                output = model(input_data.x, input_data.edge_index, input_data.batch)
            elif args.model_type == "gin":
                # GIN requires edge index and node features
                output = model(input_data.x, input_data.edge_index, input_data.batch)
            else:
                raise ValueError(f"Model type '{args.model_type}' is not supported for inference.")

            # Apply sigmoid to get probabilities for each class
            probabilities = torch.sigmoid(output)

            # Use threshold to get predicted labels (multi-label)
            predicted = (probabilities > 0.5)
            if predicted.sum() == 0:
                print("⚠️ No classes predicted above the threshold of 0.5. Using argmax as fallback.")
                predicted = torch.argmax(probabilities, dim=1, keepdim=True)  # Fallback to argmax if no classes are predicted
            print(f"✅ Inference completed successfully")

            # For each sample in the batch (usually batch size 1 in inference)
            for i in range(predicted.shape[0]):
                predicted_indices = torch.where(predicted[i])[0].cpu().numpy()
                print(f"Predicted classes for sample {i}:")
                for idx in predicted_indices:
                    if isinstance(idx, np.ndarray):
                        idx = idx.item()
                    class_name = TARGETS_LIST[int(idx)]
                    print(f" - {class_name}")
                    
        except Exception as e:
            raise RuntimeError(f"❌ Failed during model inference: {e}")

    # Interpret and display the result
    print("\n🔍 Inference Results:")
    print(f"Input SMILES: {input_smiles}")
    print(f"Predicted classes:")
    for idx in predicted_indices:
        class_name = TARGETS_LIST[int(idx)]
        print(f" - {class_name} (Probability: {probabilities[0][int(idx)].item():.4f})")
    print("="*60)

    # Ask if user wants to try another molecule
    if not args.input_smiles:  # Only ask if running interactively
        while True:
            another = input("\n🔄 Do you want to classify another molecule? (y/n): ").strip().lower()
            if another in ['y', 'yes']:
                print("\n" + "-"*60)
                new_smiles = get_smiles_input()
                # Recursive call with new SMILES
                args.input_smiles = new_smiles
                main(args)
                break
            elif another in ['n', 'no']:
                print("\n👋 Goodbye!")
                break
            else:
                print("❌ Invalid response. Please enter 'y' for yes or 'n' for no.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run inference on molecules using a trained GINESTRA model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required arguments
    parser.add_argument("--model_type", type=str, required=False, 
                       choices=["gcn", "gat", "gin", "gine", "gate", "mlp"], 
                       help="Model architecture to use.", default="gin")
    parser.add_argument("--model_path", type=str, required=False, 
                       help="Path to the trained .pt model file.", default="/repo/corradini/ginestra25/experiments/gin_MULTILABEL_BCDEF_pathway_20250808_125651/models/best_model_config_1_run_1.pt")

    # Optional arguments
    parser.add_argument("--input_smiles", type=str, required=False, 
                       help="SMILES string of the molecule to classify (optional - will prompt if not provided).", default="CCOc1ccc(cc1)C(C)C(=O)O")
    parser.add_argument("--top_k", type=int, default=1, 
                       help="Show top-k predictions.")

    args = parser.parse_args()

    # Validate arguments
    if args.top_k < 1:
        raise ValueError("top_k must be at least 1")

    try:
        main(args)
    except KeyboardInterrupt:
        print("\n\n👋 User interruption. Goodbye!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        exit(1)
