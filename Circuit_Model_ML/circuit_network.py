# How to do PC diode?  
# Hard to do via pyg.nn.MessagePassing
# just manually add or subtract then from the nodes

from PV_Circuit_Model.circuit_model import *
from PV_Circuit_Model.cell import *
from PV_Circuit_Model.multi_junction_cell import *
import torch
import torch_geometric as pyg
import torch_scatter
import pickle
import math
import numpy as np

class PC_CurrentSource(CurrentSource):
    def specify_ref_PC_diode(self,PC_diode):
        self.ref_PC_diode = PC_diode

def insert_PC_currentsource(cell:MultiJunctionCell):
    PC_diodes = []
    for subcell in reversed(cell.cells):
        if len(PC_diodes) > 0:
            pc_CurrentSource = PC_CurrentSource(IL=0.0)
            pc_CurrentSource.specify_ref_PC_diode(PC_diodes[0])
            subcell.diode_branch.add_element(pc_CurrentSource)
        PC_diodes = subcell.findElementType(PhotonCouplingDiode)

class COO():
    def __init__(self):
        self.COO_list = []
    def insert(self,index1,index2,element:CircuitElement):
        self.COO_list.append([index1,index2,element])
    @staticmethod
    def generate_edge_feature(element:CircuitElement, neg_node, pos_node):
        area = 1.0
        if element.parent is not None and isinstance(element.parent,Cell):
            area = element.parent.area
        # IL, cond, log_I0, n, breakdownV
        edge_feature = torch.zeros(1,5)
        edge_feature[0,3] = 1.0
        # diode neg node, diode pos node, corresponding neg node, corresponding pos node
        polarity_nodes = torch.tensor([-1,-1,-1,-1], dtype=torch.long)
        if isinstance(element,PC_CurrentSource):
            ref_PC_diode = element.ref_PC_diode
            edge_feature[0,2] = np.log(ref_PC_diode.I0*area)
            edge_feature[0,3] = ref_PC_diode.n
            edge_feature[0,4] = ref_PC_diode.V_shift
            polarity_nodes = torch.tensor([ref_PC_diode.neg_node,ref_PC_diode.pos_node,pos_node,neg_node], dtype=torch.long)
        elif isinstance(element,CurrentSource):
            edge_feature[0,0] = element.IL*area
            polarity_nodes = torch.tensor([-1,-1,neg_node,pos_node], dtype=torch.long)
        elif isinstance(element,Resistor):
            edge_feature[0,1] = element.cond*area
        elif isinstance(element,Diode):
            edge_feature[0,2] = np.log(element.I0*area)
            edge_feature[0,3] = element.n
            edge_feature[0,4] = element.V_shift
            polarity_nodes = torch.tensor([neg_node, pos_node, neg_node, pos_node], dtype=torch.long)
            if isinstance(element,ReverseDiode):
                polarity_nodes = torch.tensor([pos_node, neg_node, pos_node, neg_node], dtype=torch.long)
        return edge_feature, polarity_nodes
    def generate_COO_tensor(self):
        COO_tensor = torch.zeros(2*len(self.COO_list),7)
        polarity_nodes_tensor = torch.zeros(2*len(self.COO_list),4,dtype=torch.long)
        for i, _ in enumerate(self.COO_list):
            COO_tensor[i,0:2] = torch.tensor(self.COO_list[i][0:2], dtype=COO_tensor.dtype)
            edge_feature, polarity_nodes = self.generate_edge_feature(self.COO_list[i][-1], COO_tensor[i,0].item(), COO_tensor[i,1].item())
            polarity_nodes_tensor[i,:] = polarity_nodes
            COO_tensor[i,2:] = edge_feature
            COO_tensor[i+len(self.COO_list),0:2] = torch.tensor(self.COO_list[i][0:2][::-1], dtype=COO_tensor.dtype)
            COO_tensor[i+len(self.COO_list),2:] = edge_feature
            polarity_nodes_tensor[i+len(self.COO_list),:] = polarity_nodes
        return COO_tensor, polarity_nodes_tensor
    
def assign_nodes(circuit_group,node_count=0):
    if node_count==0:
        circuit_group.neg_node = node_count
        circuit_group.pos_node = node_count + 1
        node_count += 2
    if isinstance(circuit_group,CircuitGroup):
        for i, element in enumerate(circuit_group.subgroups):
            if i==0 or circuit_group.connection == "parallel":
                element.neg_node = circuit_group.neg_node
            else:
                element.neg_node = node_count 
                node_count += 1
            if i==len(circuit_group.subgroups)-1 or circuit_group.connection == "parallel":
                element.pos_node = circuit_group.pos_node
            else:
                element.pos_node = node_count
                # do not increment node count because next element neg node uses it
        for element in circuit_group.subgroups:
            node_count = assign_nodes(element,node_count=node_count)
    return node_count

def translate_to_COO(circuit_group):
    coo = COO()
    assign_nodes(circuit_group)
    circuit_elements = circuit_group.findElementType(CircuitElement)            
    for element in circuit_elements:
        coo.insert(element.neg_node,element.pos_node,element)
    COO_tensor, polarity_nodes_tensor = coo.generate_COO_tensor()
    return COO_tensor, polarity_nodes_tensor
        
class CircuitNetwork(pyg.nn.MessagePassing):
    def forward(self, data, x=None):
        if x is None:
            x = data.x
        return self.propagate(data.edge_index, x=(x,x), edge_feature=data.edge_attr, y=data.y, polarity_nodes=data.polarity_nodes_tensor, ref_x=x)

    def message(self, x_i, x_j, edge_feature, polarity_nodes, ref_x):
        # resistor - current is positive from j --> i
        cond = edge_feature[:,1].unsqueeze(1)
        I = cond*(x_j - x_i)

        # current source
        polarity = torch.ones_like(x_j)
        find_ = torch.where((polarity_nodes[:,2]>=0) & (polarity_nodes[:,2]==x_i))[0]
        polarity[find_] = -1.0
        IL = edge_feature[:,0].unsqueeze(1)
        I += polarity*IL

        # diode
        find_ = torch.where(polarity_nodes[:,0]>=0)[0]
        diode_V_drop = (ref_x[polarity_nodes[find_,1],0] - ref_x[polarity_nodes[find_,0],0]).unsqueeze(1)
        log_I0 = edge_feature[find_,2].unsqueeze(1)
        n = edge_feature[find_,3].unsqueeze(1)
        breakdownV = edge_feature[find_,4].unsqueeze(1)
        I[find_] -= polarity[find_]*torch.exp(log_I0)*(torch.exp((diode_V_drop-breakdownV)/(n*0.02568))-1.0)

        return I
    
    def aggregate(self, inputs, index, dim_size=None, y=None):
        net_I = torch_scatter.scatter(inputs, index, dim=self.node_dim, dim_size=dim_size, reduce="sum")
        node_error = net_I
        if y is not None:
            indices = torch.where(y[:,0]==1) # pinned voltage boundary condition
            node_error[indices] = 0.0 # just assume the voltage pinned nodes are set at the desired voltage, we don't measure voltage errors
            indices = torch.where(y[:,0]==2) # pinned current boundary condition
            node_error[indices] = net_I[indices] - y[indices[0],2].unsqueeze(1) 
        return node_error

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

    def forward(self, x, edge_index, edge_feature, polarity_nodes):
        edge_out, aggr = self.propagate(edge_index, x=(x, x), edge_feature=edge_feature, polarity_nodes=polarity_nodes, ref_x=x)
        node_out = self.lin_node(torch.cat((x, aggr), dim=-1))
        edge_out = edge_feature + edge_out
        node_out = x + node_out
        return node_out, edge_out

    def message(self, x_i, x_j, edge_feature, polarity_nodes, ref_x):
        x_i_ = x_i
        x_j_ = x_j
        
        x = torch.cat((x_i, x_j, edge_feature), dim=-1)
        x = self.lin_edge(x)
        return x

    def aggregate(self, inputs, index, dim_size=None):
        out = torch_scatter.scatter(inputs, index, dim=self.node_dim, dim_size=dim_size, reduce="sum")
        return (inputs, out)
    
class LearnedSimulator(torch.nn.Module):
    def __init__(
        self,
        hidden_size=32,
        n_mp_layers=4, # number of GNN layers
        embedding_dim=8
    ):
        super().__init__()
        self.embed_type = torch.nn.Embedding(3, embedding_dim)
        self.node_in = MLP(embedding_dim + 2, hidden_size, hidden_size, 3)
        self.edge_in = MLP(8, hidden_size, hidden_size, 3)
        self.node_out = MLP(hidden_size, hidden_size, 1, 3, layernorm=False)
        self.n_mp_layers = n_mp_layers
        self.layers = torch.nn.ModuleList([InteractionNetwork(
            hidden_size, 3
        ) for _ in range(n_mp_layers)])
        self.cn = CircuitNetwork()

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.embed_type.weight)

    def forward(self, data: pyg.data.Data, modifiers = None):
        # pre-processing
        # node feature: combine categorial feature data.x and contiguous feature data.pos.
        node_feature = torch.cat((self.embed_type(data.x[:,0]), data.x[:,1:]), dim=-1)
        node_feature = self.node_in(node_feature)
        edge_feature = self.edge_in(data.edge_attr)
        # stack of GNN layers
        for i in range(self.n_mp_layers):
            node_feature, edge_feature = self.layers[i](node_feature, data.edge_index, edge_feature=edge_feature)
        # post-processing
        out = self.node_out(node_feature)

        # indices = torch.where(data.x[:,0]==1)[0] # pinned voltage boundary condition
        # out[indices,0] = data.x[indices,1].float()
        node_error = self.cn.forward(data, out)

        return out, node_error


if __name__ == "__main__":
    tandem_model = pickle.load(open(r"C:\Users\arson\Documents\Tandem_Cell_Fit_Tools\best_fit_tandem_model.pkl", 'rb'))
    insert_PC_currentsource(tandem_model)

    tandem_model_clone = circuit_deepcopy(tandem_model)

    target_I = 0
    tandem_model.set_operating_point(I=target_I)
    V1 = tandem_model.operating_point[0]
    V3 = V1-tandem_model.subgroups[2].operating_point[0]
    V2 = V3-tandem_model.subgroups[1].operating_point[0]
    assign_nodes(tandem_model)
    tandem_model.draw(display_value=True)
        
    coo, polarity_nodes_tensor = translate_to_COO(tandem_model)
    edge_index = coo[:,:2].long().t()
    edge_feature = coo[:,2:]
    max_node_index = edge_index.max().item()

    x = torch.tensor([0.0,V1,V2,V3])[:,None]
    print(x)
    # 0th index is the boundary condition:
    # 0-not a boundary; 1-voltage pinning (Dirichlet boundary condition); 2-net current pinning
    # 1st index is the voltage to pin (if 0th index is 1)
    # 2nd index is the net current to pin (if 0th index is 2)
    node_boundary_conditions = torch.tensor([[1,0,0],[2,0,target_I],[0,0,0],[0,0,0]],dtype=torch.float)
    cn = CircuitNetwork()

    data = pyg.data.Data(
        x=x,  
        edge_index=edge_index,  
        edge_attr=edge_feature, 
        y=node_boundary_conditions,
        polarity_nodes_tensor = polarity_nodes_tensor
    )

    node_error = cn.forward(data)
    rows = torch.where(node_boundary_conditions[:,0] != 1)[0]

    for i in range(5):
        J = torch.autograd.functional.jacobian(lambda x_: cn(pyg.data.Data(x=x_, edge_index=edge_index, edge_attr=edge_feature, polarity_nodes_tensor=polarity_nodes_tensor)), x).squeeze()
        J = J[rows][:, rows]

        node_error = node_error[rows]
        Y = -node_error

        X = torch.linalg.solve(J, Y)

        x[rows] = x[rows] + X

        data = pyg.data.Data(
            x=x, 
            edge_index=edge_index,  
            edge_attr=edge_feature,
            y=node_boundary_conditions,
            polarity_nodes_tensor = polarity_nodes_tensor
        )

        node_error = cn.forward(data)
    
    print(x)
    print(node_error)
