import torch
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
from PIL import Image
import io, os
import matplotlib.cm as cm
import time 
from torch_geometric.explain import Explainer, GNNExplainer
from torch_geometric.explain.config import ModelConfig

# Le tue funzioni (con piccole correzioni)
def pyg_data_to_mol(data, permitted_list_of_atoms):
    """Converts PyG Data object to RDKit Mol object
    Args:
        data (torch_geometric.data.Data): PyG Data object containing node features and edge indices.
        permitted_list_of_atoms (list): List of permitted atom symbols.
    Returns:
        rdkit.Chem.rdchem.Mol: RDKit Mol object.
    """
    mol = Chem.RWMol()

    # Aggiungi atomi
    for i in range(data.num_nodes):
        idx = int(data.x[i][0].item()*len(permitted_list_of_atoms))  # ID intero dell'atomo
        if idx >= len(permitted_list_of_atoms):
            raise ValueError(f"Atom index {idx} out of bounds for permitted list.")
        symbol = permitted_list_of_atoms[idx]
        mol.AddAtom(Chem.Atom(symbol))

    # Aggiungi legami
    for k in range(data.edge_index.size(1)):
        i = int(data.edge_index[0, k].item())
        j = int(data.edge_index[1, k].item())

        # Evita doppie aggiunte se bidirezionale
        if i < j:
            bond_type = Chem.BondType.SINGLE  # default
            if hasattr(data, 'edge_attr') and data.edge_attr is not None:
                bond_val = int(data.edge_attr[k][0].item())
                bond_type = {
                    1: Chem.BondType.SINGLE,
                    2: Chem.BondType.DOUBLE,
                    3: Chem.BondType.TRIPLE,
                    4: Chem.BondType.AROMATIC
                }.get(bond_val, Chem.BondType.SINGLE)
            mol.AddBond(i, j, bond_type)

    # Finalizzazione
    mol = mol.GetMol()
    Chem.SanitizeMol(mol)
    return mol


def draw_molecule_with_explanation(
    mol,    
    edge_index,
    node_importance,
    edge_importance=None,
    predicted_class=0,
    save_path="explained_molecule.png",
    dpi=300,
    cmap_name="Reds",
    smile=None
):
    """Draws a molecule with node and edge importance using RDKit and saves the image.
    Args:
        mol (rdkit.Chem.rdchem.Mol): RDKit Mol object.
        edge_index (torch.Tensor): Edge index tensor of shape [2, num_edges].
        node_importance (list): List of node importance values.
        edge_importance (list, optional): List of edge importance values. Defaults to None.
        predicted_class (int, optional): Predicted class for the molecule. Defaults to 0.
        save_path (str, optional): Path to save the image. Defaults to "explained_molecule.png".
        dpi (int, optional): DPI for the saved image. Defaults to 300.
        cmap_name (str, optional): Colormap name for highlighting. Defaults to "Reds".
        smile (str, optional): SMILES representation of the molecule for visualization. Defaults to None.
    """
    
    # Normalizza importanza nodi
    if len(node_importance) > 0:
        max_node = max(node_importance) if max(node_importance) > 0 else 1.0
        min_node = min(node_importance)
        norm_node = [(x - min_node) / (max_node - min_node + 1e-8) for x in node_importance]
    else:
        norm_node = [0.0] * mol.GetNumAtoms()
    
    cmap = cm.get_cmap(cmap_name)
    
    # Colori atomi basati su importanza
    atom_colors = {}
    for i, importance in enumerate(norm_node):
        if importance > 0.3:  # Soglia per evidenziare
            rgba = cmap(importance)
            atom_colors[i] = rgba
    
    # Colori legami se disponibili
    bond_colors = {}
    if edge_importance is not None:
        max_edge = max(edge_importance) if len(edge_importance) > 0 and max(edge_importance) > 0 else 1.0
        min_edge = min(edge_importance) if len(edge_importance) > 0 else 0.0
        
        edge_dict = {}
        for k, (i, j) in enumerate(edge_index.t().tolist()):
            if k < len(edge_importance):
                key = tuple(sorted((i, j)))
                edge_dict[key] = max(edge_dict.get(key, 0), edge_importance[k])
        
        for bond in mol.GetBonds():
            a1 = bond.GetBeginAtomIdx()
            a2 = bond.GetEndAtomIdx()
            key = tuple(sorted((a1, a2)))
            if key in edge_dict:
                norm_edge = (edge_dict[key] - min_edge) / (max_edge - min_edge + 1e-8)
                if norm_edge > 0.3:  # Soglia
                    rgba = cmap(norm_edge)
                    bond_colors[bond.GetIdx()] = rgba

    # Disegna molecola
    mol_copy = Chem.Mol(mol)
    Chem.rdDepictor.Compute2DCoords(mol_copy)
    drawer = rdMolDraw2D.MolDraw2DCairo(1200, 1200)
    drawer.drawOptions().addAtomIndices = True
    
    drawer.DrawMolecule(
        mol_copy,
        highlightAtoms=list(atom_colors.keys()),
        highlightBonds=list(bond_colors.keys()),
        highlightAtomColors=atom_colors,
        highlightBondColors=bond_colors,
    )
    drawer.FinishDrawing()
    mol_img = Image.open(io.BytesIO(drawer.GetDrawingText()))

    # Colorbar
    fig, ax = plt.subplots(figsize=(0.35, 4), dpi=dpi)
    norm = plt.Normalize(vmin=0.0, vmax=1.0)
    smap = cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = plt.colorbar(smap, cax=ax, ticks=[0.0, 0.25, 0.5, 0.75, 1.0])
    cb.ax.tick_params(labelsize=7, length=3, direction='in', pad=2)
    cb.set_label(f"Importance (Class {predicted_class})", fontsize=8, labelpad=5)

    fig.subplots_adjust(left=0.2, right=0.8)
    fig.canvas.draw()
    colorbar_img = Image.frombytes("RGB", fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
    plt.close(fig)

    # Combina immagini
    colorbar_img = colorbar_img.resize((colorbar_img.width, mol_img.height))
    total_width = mol_img.width + colorbar_img.width
    result = Image.new("RGB", (total_width, mol_img.height), color=(255, 255, 255))
    result.paste(mol_img, (0, 0))
    result.paste(colorbar_img, (mol_img.width, 0))
    
    plt.figure(figsize=(12, 8))
    plt.subplot(1, 2, 1)
    if smile:
        mol_from_smile = Chem.MolFromSmiles(smile)
        if mol_from_smile:
            Chem.rdDepictor.Compute2DCoords(mol_from_smile)
            img_smile = Chem.Draw.MolToImage(mol_from_smile, size=(400, 400))
            plt.imshow(img_smile)
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(result)
    plt.axis('off')
    plt.show()
    plt.savefig(save_path.replace('.png', '_full.png'), dpi=dpi, bbox_inches='tight')
    print(f"✅ Explained molecule saved: {save_path.replace('.png', '_full.png')}")
    
    return result


def explain_molecule_with_rdkit(explainer, single_data, predicted_class, 
                               permitted_atoms, true_class=None, smile=None, output_dir="explanations"):
    
    explanation = explainer(
        x=single_data.x,
        edge_index=single_data.edge_index,
        batch=single_data.batch,
        target=predicted_class
    )
    
    node_mask = explanation.node_mask.cpu().detach().numpy()
    if len(node_mask.shape) > 1:
        node_importance = node_mask.sum(axis=1)
    else:
        node_importance = node_mask
    
    edge_importance = None
    try:
        edge_importance = explanation.edge_mask.cpu().detach().numpy()
    except:
        print("Edge importance not available in this explanation.")
    
    print(f"\n{'='*60}")
    print(f"ANALISI MOLECOLA - CLASSE PREDETTA: {predicted_class}")
    if true_class is not None:
        print(f"CLASSE VERA: {true_class}")
        print(f"PREDIZIONE CORRETTA: {'✅' if predicted_class == true_class else '❌'}")
    print(f"{'='*60}")
    
    print(f"Numero atomi: {len(node_importance)}")
    print(f"Importanza media atomi: {node_importance.mean():.4f}")
    print(f"Importanza max: {node_importance.max():.4f}")
    
    top_atoms = node_importance.argsort()[-5:][::-1]
    print(f"Top 5 atomi più importanti: {top_atoms}")
    print(f"Loro importanza: {[f'{node_importance[i]:.4f}' for i in top_atoms]}")
    
    if edge_importance is not None:
        top_edges = edge_importance.argsort()[-5:][::-1]
        print(f"Top 5 legami più importanti: {top_edges}")
        print(f"Loro importanza: {[f'{edge_importance[i]:.4f}' for i in top_edges]}")
    
    try:
        mol = pyg_data_to_mol(single_data, permitted_atoms)
        print(f"✅ Molecola convertita: {mol.GetNumAtoms()} atomi, {mol.GetNumBonds()} legami")
        
        timestamp = time.strftime("%H%M%S")
        # Visualizza con RDKit
        draw_molecule_with_explanation(
            mol=mol,
            edge_index=single_data.edge_index,
            node_importance=node_importance,
            edge_importance=edge_importance,
            predicted_class=predicted_class,
            save_path=os.path.join(output_dir, f"explanation_{predicted_class}_{timestamp}.png") if output_dir else "explanation.png",
            smile=smile,  
        )
        
    except Exception as e:
        print(f"❌ Errore conversione RDKit: {e}")
    
    return explanation


def explain_all_molecules_in_batches(model, test_dataloader, device, permitted_atoms, max_molecules=10, smiles_file=None, output_dir="explanations"):
    """Explains all molecules in the test dataloader in batches using GNNExplainer and RDKit visualization.
    Args:
        model (torch.nn.Module): The trained GNN model.
        test_dataloader (torch_geometric.data.DataLoader): DataLoader for the test dataset.
        device (torch.device): Device to run the model on (CPU or GPU).
        permitted_atoms (list): List of permitted atom symbols.
        max_molecules (int, optional): Maximum number of molecules to explain. Defaults to 10.
        smiles_file (str, optional): Path to a CSV file containing SMILES strings for the molecules. Defaults to None.
        output_dir (str, optional): Directory to save the explanation images. Defaults to "explanations".
    """
    import pandas as pd
    
    if smiles_file is not None:
        smiles_df = pd.read_csv(smiles_file)
        smiles = smiles_df['SMILES'].tolist()
        
    explainer = Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=50),
        explanation_type='model',
        node_mask_type='attributes',
        edge_mask_type='object',
        model_config=ModelConfig(
            mode='multiclass_classification',
            task_level='graph',
            return_type='probs'
        )
    )
    
    total_processed = 0
    batch_num = 0
    
    for batch in test_dataloader:
        batch_num += 1
        batch = batch.to(device)
        
        unique_batch_ids = torch.unique(batch.batch)
        print(f"\n🔍 BATCH {batch_num}: {len(unique_batch_ids)} molecules")
        
        for i, graph_idx in enumerate(unique_batch_ids):
            if total_processed >= max_molecules:
                print(f"Maximum number of molecules ({max_molecules}) reached. Stopping analysis.")
                return
                
            graph_idx = graph_idx.item()
            
            try:
                # Graph extraction
                mask = (batch.batch == graph_idx)
                single_x = batch.x[mask]
                single_y = batch.y[graph_idx]
                
                node_indices = torch.where(mask)[0]
                edge_mask = torch.isin(batch.edge_index[0], node_indices) & torch.isin(batch.edge_index[1], node_indices)
                single_edge_index = batch.edge_index[:, edge_mask]
                
                node_map = {old.item(): new for new, old in enumerate(node_indices)}
                single_edge_index = torch.tensor([[node_map[idx.item()] for idx in row] for row in single_edge_index])
                single_edge_index = single_edge_index.cuda()
                
                single_batch = torch.zeros(single_x.size(0), dtype=torch.long).to(device)
                single_x.requires_grad_(True)
                
                single_data = type('Data', (), {
                    'x': single_x,
                    'edge_index': single_edge_index,
                    'batch': single_batch,
                    'num_nodes': single_x.size(0)
                })()
                
                # Prediction
                kwargs = {}
                if hasattr(batch, 'fingerprint_tensor'):
                    kwargs['fingerprint'] = batch.fingerprint_tensor[graph_idx:graph_idx+1]
                
                out = model(single_x, single_edge_index, single_batch, **kwargs)
                predicted_class = out.argmax(dim=1).item()
                
                print(f"\n{'='*60}")
                print(f"MOLECULE {total_processed + 1}")
                print(f"Batch {batch_num}, Graph {i+1}/{len(unique_batch_ids)}")
                print(f"Atoms: {single_x.size(0)}, Bonds: {single_edge_index.size(1)}")
                print(f"Predicted class: {predicted_class}")
                print(f"{'='*60}")
                
                explanation = explain_molecule_with_rdkit(
                    explainer=explainer,
                    single_data=single_data,
                    predicted_class=predicted_class,
                    permitted_atoms=permitted_atoms,
                    true_class=single_y.argmax(dim=0).item() if single_y is not None else None,
                    smile=smiles[total_processed] if smiles_file else None,
                    output_dir=output_dir
                )
                
                total_processed += 1
                
            except Exception as e:
                print(f"❌ Error molecule {total_processed + 1}: {e}")
                continue
        
        print(f"📊 Batch {batch_num} completed: {len(unique_batch_ids)} molecules processed")

    print(f"✅ Total molecules processed: {total_processed}")