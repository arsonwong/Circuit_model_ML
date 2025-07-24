import torch
import torch_geometric as pyg
import torch_scatter
import math
from .utilities import *
from .circuit_network import *

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
    def __init__(self, hidden_size, layers):
        super().__init__()
        self.lin_edge = MLP(hidden_size * 3, hidden_size, hidden_size, layers)
        self.lin_node = MLP(hidden_size * 2, hidden_size, hidden_size, layers)

    def forward(self, x, edge_index, edge_feature, diode_nodes):
        edge_out, aggr = self.propagate(edge_index, x=(x, x), edge_feature=edge_feature, diode_nodes=diode_nodes, ref_x=x)
        node_out = self.lin_node(torch.cat((x, aggr), dim=-1))
        edge_out = edge_feature + edge_out
        node_out = x + node_out
        return node_out, edge_out

    def message(self, x_i, x_j, edge_feature, diode_nodes, ref_x):
        x_i_ = x_i
        x_j_ = x_j
        find_ = torch.where(diode_nodes[:,0]>=0)[0]
        x_i_[find_] = ref_x[diode_nodes[find_,0]]
        x_j_[find_] = ref_x[diode_nodes[find_,1]]
        x = torch.cat((x_i_, x_j_, edge_feature), dim=-1)
        x = self.lin_edge(x)
        return x

    def aggregate(self, inputs, index, dim_size=None):
        out = torch_scatter.scatter(inputs, index, dim=self.node_dim, dim_size=dim_size, reduce="sum")
        return (inputs, out)

# nodes: put in guess voltage, net I as well
# guess the direction, so layer norm can be True    
class LearnedSimulator(torch.nn.Module):
    def __init__(
        self,
        hidden_size=64,
        n_mp_layers=4, # number of GNN layers
        embedding_dim=8
    ):
        super().__init__()
        self.embed_type = torch.nn.Embedding(3, embedding_dim)
        self.node_in = MLP(embedding_dim + 4, hidden_size, hidden_size, 3)
        self.edge_in = MLP(5, hidden_size, hidden_size, 3)
        self.node_out = MLP(hidden_size, hidden_size, 1, 3)
        self.n_mp_layers = n_mp_layers
        self.layers = torch.nn.ModuleList([InteractionNetwork(
            hidden_size, 3
        ) for _ in range(n_mp_layers)])
        self.cn = CircuitNetwork(max_log_diode_I = 4)

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.embed_type.weight)

    def forward(self, data: pyg.data.Data, modifiers = None):
        node_error = self.cn.forward(data)

        node_feature = torch.cat((self.embed_type(data.y[:,0].long()), data.y[:,1:], data.x, node_error), dim=-1)
        node_feature = self.node_in(node_feature)
        edge_feature = self.edge_in(data.edge_attr)
        # stack of GNN layers
        for i in range(self.n_mp_layers):
            node_feature, edge_feature = self.layers[i](node_feature, data.edge_index, edge_feature=edge_feature, diode_nodes=data.diode_nodes_tensor)
        # post-processing
        out = self.node_out(node_feature)

        return out, node_error

