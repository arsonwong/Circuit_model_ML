import torch
import torch_geometric as pyg
from torch_geometric.data import Batch
import sys, os
sys.path.insert(0, os.path.abspath(r"D:\Griddler\physicsnemo"))
sys.path.insert(1,os.path.abspath(r"D:\Griddler\BSMS-GNN\src"))

from graph_wrappers.bsms_graph_wrapper import BistrideMultiLayerGraph
from graph_wrappers.graph_wrapper import GraphType
import numpy as np
import torch

from Circuit_Model_ML.circuit_network_ML import GridCircuit  
from physicsnemo.models.meshgraphnet.bsms_mgn import BiStrideMeshGraphNet  

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# for the bsms collate fn we will format the batch to pyG with these fields
# node_features,
#         edge_features,
#         graph=data,
#         ms_edges=ms_edges,
#         ms_ids=ms_ids,

def bsms_collate_fn(data_list):
    # Standard PyG batching (x, edge_attr, y, etc.)
    batch = Batch.from_data_list(data_list)

    # -------- 2) Batch BSMS multilevel structures (ms_edges, ms_ids) --------
    # Assume all samples have same number of levels
    num_levels = len(data_list[0].ms_edges) - 1  # num_mesh_levels

    # Per-level container for edges/ids after batching
    batched_ms_edges = [[] for _ in range(num_levels + 1)]  # levels 0..L
    batched_ms_ids   = [[] for _ in range(num_levels)]      # levels 0..L-1

    # Per-level node offsets (like node_offset_fine but per BSMS level)
    level_offsets = [0] * (num_levels + 1)

    for data in data_list:
        # Node counts per level for this sample
        # level 0 is fine level = data.num_nodes
        N_l = [data.num_nodes]
        # level l>0 has len(ms_ids[l-1]) nodes, because those are the kept indices
        for l in range(num_levels):
            N_l.append(int(data.ms_ids[l].shape[0]))

        # Save starting offset for this sample at each level
        base_offsets = level_offsets.copy()

        # --- edges per level ---
        for l in range(num_levels + 1):
            # local edges for this sample at level l
            e_l = data.ms_edges[l]  # shape [2, E_l_sample], indices 0..N_l[l]-1
            # shift to global indices for this level
            e_l_global = e_l + base_offsets[l]
            batched_ms_edges[l].append(e_l_global)

            # advance offset at this level
            level_offsets[l] += N_l[l]

        # --- ids per level ---
        for l in range(num_levels):
            # local ids at this level: indices into level-l nodes (0..N_l[l]-1)
            ids_l = data.ms_ids[l]
            # shift to global indices into the big level-l node array
            ids_l_global = ids_l + base_offsets[l]
            batched_ms_ids[l].append(ids_l_global)

    # Concatenate per-level results
    batch.ms_edges = [torch.cat(e_list, dim=1) for e_list in batched_ms_edges]
    batch.ms_ids   = [torch.cat(i_list, dim=0) for i_list in batched_ms_ids]

    return batch



def build_circuit_data(rows=4, cols=4, is_linear=False, has_current_source=False):
    grid = GridCircuit(
        rows=rows,
        cols=cols,
        is_linear=is_linear,
        has_current_source=has_current_source,
    )
    data: pyg.data.Data = grid.export()
    # Add pos, but do not actually include positional information
    data.pos = torch.zeros((grid.num_nodes, 2), dtype=torch.float32)
    return grid, data


def build_bistride_model(node_dim: int, edge_dim: int) -> BiStrideMeshGraphNet:
    model = BiStrideMeshGraphNet(
        input_dim_nodes=node_dim,
        input_dim_edges=edge_dim,
        output_dim=1,          # predict ΔV per node
        processor_size=8,
        num_mesh_levels=1,     # single mesh level for now
        bistride_pos_dim=2,    # we passed 2D positions (x, y)
        bistride_unet_levels=1,
        hidden_dim_processor=128,
    )
    return model.to(device)


def example_forward():
    # ---- Build one random circuit graph ----
    grid, data = build_circuit_data(rows=5, cols=5, is_linear=False, has_current_source=True)

    data = data.to(device)


    edge_index = data.edge_index.cpu().numpy()  # shape [2, E]

    # 1) Make sure edges are undirected (BSMS code expects that)
    flat_edges = np.concatenate([edge_index, edge_index[::-1, :]], axis=1)

    # 2) Build positions for BSMS (use your grid layout)
    #    For example, 2D positions plus a dummy zero z:
    node_x = grid.node_cols   # or however you store col indices
    node_y = grid.node_rows   # row indices
    pos_mesh = np.stack([node_x, node_y, np.zeros_like(node_x)], axis=-1).astype(np.float32)
    num_nodes = data.num_nodes
    # 3) Build the multiscale graphs - move out of forward!  you only need to do this once, not every forward pass
    num_layers = 1  # number of coarse levels (matches num_mesh_levels)
    multi = BistrideMultiLayerGraph(flat_edges, num_layers, num_nodes, pos_mesh)
    m_gs, m_flat_es, m_ids = multi.get_multi_layer_graphs()

    ms_edges = [torch.from_numpy(fe).long().to(device) for fe in m_flat_es]  # length = num_layers+1
    ms_ids   = [torch.from_numpy(ids).long().to(device) for ids in m_ids]    # length = num_layers

    # ---- Build node and edge features ----
    # y[:,0]: BC type (0=none, 1=voltage pin, 2=current pin)
    # y[:,1]: BC voltage (if type==1)
    # x: current voltage guess
    node_bc_type = data.y[:, 0:1]   # (N,1)
    node_bc_val  = data.y[:, 1:2]   # (N,1)
    node_guess   = data.x           # (N,1) initial voltage

    # Simple node feature: [BC_type, BC_value, V_guess]
    node_features = torch.cat([node_bc_type, node_bc_val, node_guess], dim=-1).float()

    # Edge features are already in the right format: [IL, cond, log_I0, n, breakdownV]
    edge_features = data.edge_attr.float()

    # ---- Build and run the BiStrideMeshGraphNet ----
    model = build_bistride_model(
        node_dim=node_features.shape[1],
        edge_dim=edge_features.shape[1],
    )

    model = model.to(device)

    # Forward: graph is the PyG Data (needs .edge_index and .pos)
    dV_pred = model(
        node_features,
        edge_features,
        graph=data,
        ms_edges=ms_edges,
        ms_ids=ms_ids,
    )

    # Interpret output as voltage update
    V_pred = node_guess + dV_pred

    print("Predicted node voltages shape:", V_pred.shape)
    print("First few V_pred:", V_pred[:5].view(-1).detach().cpu().numpy())


if __name__ == "__main__":
    example_forward()
