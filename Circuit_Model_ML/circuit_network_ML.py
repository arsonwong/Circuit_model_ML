import torch
import torch_geometric as pyg
import torch_scatter
import math
from .utilities import *
from .circuit_network import *
import os, sys
sys.path.insert(0, os.path.abspath(r"D:\Griddler\physicsnemo"))
from physicsnemo.models.meshgraphnet.bsms_mgn import BiStrideMeshGraphNet 
sys.path.insert(1,os.path.abspath(r"D:\Griddler\BSMS-GNN\src"))
from graph_wrappers.bsms_graph_wrapper import BistrideMultiLayerGraph
from graph_wrappers.graph_wrapper import GraphType
import numpy as np

class MLP(torch.nn.Module):
    """Multi-Layer perceptron"""
    def __init__(self, input_size, hidden_size, output_size, layers, layernorm=True):
        super().__init__()
        self.layers = torch.nn.ModuleList()
        self.input_size = input_size
        for i in range(layers):
            self.layers.append(torch.nn.Linear(
                input_size if i == 0 else hidden_size,
                output_size if i == layers - 1 else hidden_size,
            ))
            if i != layers - 1:
                self.layers.append(torch.nn.ReLU())
        if layernorm and output_size>1:
            self.layers.append(torch.nn.LayerNorm(output_size))
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.layers:
            if isinstance(layer, torch.nn.Linear):
                layer.weight.data.normal_(0, 1 / math.sqrt(layer.in_features))
                layer.bias.data.fill_(0)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
class InteractionNetwork(pyg.nn.MessagePassing):
    def __init__(self, hidden_size, layers, layernorm):
        super().__init__()
        self.lin_edge = MLP(hidden_size * 3, hidden_size, hidden_size, layers, layernorm=layernorm)
        self.lin_node = MLP(hidden_size * 2, hidden_size, hidden_size, layers, layernorm=layernorm)

    def forward(self, x, edge_index, edge_feature):
        edge_out, aggr = self.propagate(edge_index, x=(x, x), edge_feature=edge_feature)
        node_out = self.lin_node(torch.cat((x, aggr), dim=-1))
        edge_out = edge_feature + edge_out
        node_out = x + node_out
        return node_out, edge_out

    def message(self, x_i, x_j, edge_feature):
        x = torch.cat((x_i, x_j, edge_feature), dim=-1)
        x = self.lin_edge(x)
        return x

    def aggregate(self, inputs, index, dim_size=None):
        out = torch_scatter.scatter(inputs, index, dim=self.node_dim, dim_size=dim_size, reduce="sum")
        return (inputs, out)

 
class GNN(torch.nn.Module):
    def __init__(
        self,
        input_dim_nodes=4,
        input_dim_edges=4,
        hidden_size=64,
        n_mp_layers=4, # number of GNN layers
        layernorm=True
    ):
        super().__init__()
        self.node_in = MLP(input_dim_nodes, hidden_size, hidden_size, 3, layernorm=layernorm)
        self.edge_in = MLP(input_dim_edges, hidden_size, hidden_size, 3, layernorm=layernorm)
        self.node_out = MLP(hidden_size, hidden_size, 1, 3, layernorm=False)
        self.n_mp_layers = n_mp_layers
        self.layers = torch.nn.ModuleList([InteractionNetwork(
            hidden_size, 3, layernorm=layernorm
        ) for _ in range(n_mp_layers)])
        self.cn = CircuitNetwork(max_log_diode_I = 4)

    def forward(self, node_features, edge_features, graph):
        node_feature = self.node_in(node_features)
        edge_feature = self.edge_in(edge_features.float())
        # stack of GNN layers
        for i in range(self.n_mp_layers):
            node_feature, edge_feature = self.layers[i](node_feature, graph.edge_index, edge_feature=edge_feature)
        # post-processing
        out = self.node_out(node_feature)
        return out


class LearnedSimulator(torch.nn.Module):
    def __init__(
        self,
        input_dim_nodes=4,
        input_dim_edges=4,
        hidden_size=64,
        n_mp_layers=4, # number of GNN layers
        layernorm=True,
        num_mesh_levels = 3,
        model_type = "gnn" # gnn or bsms
    ):
        super().__init__()
        self.num_mesh_levels = num_mesh_levels
        self.model_type = model_type
        if model_type=="bsms":
            self.model = BiStrideMeshGraphNet(
                input_dim_nodes=input_dim_nodes,
                input_dim_edges=input_dim_edges,
                output_dim=1,          # predict ΔV per node
                processor_size=n_mp_layers,
                num_mesh_levels=num_mesh_levels,     # single mesh level for now
                bistride_pos_dim=2,    # we passed 2D positions (x, y)
                hidden_dim_processor=hidden_size,
            )
        else:
            self.model = GNN(input_dim_nodes=input_dim_nodes,
                input_dim_edges=input_dim_edges,
                hidden_size=hidden_size,
                n_mp_layers=n_mp_layers,
                layernorm=layernorm)
        self.cn = CircuitNetwork(max_log_diode_I = 4)

    def forward(self, data: pyg.data.Data):
        node_error = self.cn.forward(data).float()
        node_feature = torch.cat((data.y.float(), data.x.float(), node_error), dim=-1)

        if self.model_type=="bsms":
            N = data.num_nodes
            edge_index_np = data.edge_index.detach().cpu().numpy().astype(np.int64)
            pos_mesh = np.zeros((N, 2), dtype=np.float32)
            multi = BistrideMultiLayerGraph(edge_index_np, self.num_mesh_levels, data.x.shape[0], pos_mesh)
            _, m_flat_es, m_ids = multi.get_multi_layer_graphs()

            ms_edges = [torch.from_numpy(fe).long().to(data.x.device) for fe in m_flat_es]  # length = num_layers+1
            ms_ids   = [torch.from_numpy(ids).long().to(data.x.device) for ids in m_ids]    # length = num_layers
            data_ = pyg.data.Data(edge_index = data.edge_index, pos=torch.zeros((data.num_nodes, 2),device=data.x.device,dtype=torch.float32))

            out = self.model(
                node_feature,
                data.edge_attr,
                graph=data_,
                ms_edges=ms_edges,
                ms_ids=ms_ids)
        else:
            data_ = pyg.data.Data(edge_index = data.edge_index)
            out = self.model(node_feature,data.edge_attr,graph=data_)
        return out, node_error
