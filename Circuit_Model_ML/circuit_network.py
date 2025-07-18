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
from matplotlib import pyplot as plt
from torch_geometric.utils import to_networkx
import networkx as nx
from utilities import *

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
    def generate_edge_feature(element:CircuitElement, neg_node, pos_node, polarity):
        area = 1.0
        if element.parent is not None and isinstance(element.parent,Cell):
            area = element.parent.area
        # IL, cond, log_I0, n, breakdownV
        edge_feature = torch.zeros(1,5)
        edge_feature[0,3] = 1.0
        # diode neg node, diode pos node
        diode_nodes = torch.tensor([-1,-1], dtype=torch.long)
        if isinstance(element,PC_CurrentSource):
            ref_PC_diode = element.ref_PC_diode
            edge_feature[0,2] = -polarity*np.log(ref_PC_diode.I0*area)
            edge_feature[0,3] = ref_PC_diode.n
            edge_feature[0,4] = ref_PC_diode.V_shift
            diode_nodes = torch.tensor([ref_PC_diode.neg_node,ref_PC_diode.pos_node], dtype=torch.long)
        elif isinstance(element,CurrentSource):
            edge_feature[0,0] = polarity*element.IL*area
        elif isinstance(element,Resistor):
            edge_feature[0,1] = element.cond*area
        elif isinstance(element,Diode):
            edge_feature[0,2] = polarity*np.log(element.I0*area)
            edge_feature[0,3] = element.n
            edge_feature[0,4] = element.V_shift
            diode_nodes = torch.tensor([neg_node, pos_node], dtype=torch.long)
            if isinstance(element,ReverseDiode):
                diode_nodes = torch.tensor([pos_node, neg_node], dtype=torch.long)
                edge_feature[0,2] *= -1
        return edge_feature, diode_nodes
    def generate_COO_tensor(self):
        COO_tensor = torch.zeros(2*len(self.COO_list),7)
        diode_nodes_tensor = torch.zeros(2*len(self.COO_list),2,dtype=torch.long)
        for i, _ in enumerate(self.COO_list):
            COO_tensor[i,0:2] = torch.tensor(self.COO_list[i][0:2], dtype=COO_tensor.dtype)
            edge_feature, diode_nodes = self.generate_edge_feature(self.COO_list[i][-1], COO_tensor[i,0].item(), COO_tensor[i,1].item(), polarity=1)
            COO_tensor[i,2:] = edge_feature
            diode_nodes_tensor[i,:] = diode_nodes
            COO_tensor[i+len(self.COO_list),0:2] = torch.tensor(self.COO_list[i][0:2][::-1], dtype=COO_tensor.dtype)
            edge_feature, diode_nodes = self.generate_edge_feature(self.COO_list[i][-1], COO_tensor[i,0].item(), COO_tensor[i,1].item(), polarity=-1)
            COO_tensor[i+len(self.COO_list),2:] = edge_feature
            diode_nodes_tensor[i+len(self.COO_list),:] = diode_nodes
        return COO_tensor, diode_nodes_tensor
    
class GridCircuit():
    diode_V_hard_limit = 0.8
    diode_V_pos_delta_limit = 0.5
    def __init__(self,rows=4,cols=4,is_linear=False):
        self.rows = rows
        self.cols = cols
        self.is_linear = is_linear
        self.fill_nodes()
        self.fill_edges()
        self.fill_boundary_conditions()
    def fill_nodes(self,num_nodes=None):
        max_ = self.rows*self.cols
        if num_nodes is None:
            num_nodes = int(max_ * 0.8)
        if num_nodes > max_:
            num_nodes = max_
        rand_perm = np.random.permutation(max_)
        self.num_nodes = num_nodes
        self.node_pos = rand_perm[:num_nodes]
        self.node_rows = np.floor(self.node_pos / self.cols)
        self.node_cols = self.node_pos - self.node_rows*self.cols
        has_edge = np.ones(self.num_nodes, dtype=bool)
        for i in range(self.num_nodes):
            indices = np.where(np.abs(self.node_rows-self.node_rows[i])+np.abs(self.node_cols-self.node_cols[i])<=1)[0]
            if len(indices)==1:
                has_edge[i] = False
        find_ = np.where(has_edge==True)[0]
        self.node_pos = self.node_pos[find_]
        self.node_rows = self.node_rows[find_]
        self.node_cols = self.node_cols[find_]
        self.num_nodes = len(find_)
    def draw(self):
        if hasattr(self,"node_cols"):
            _, ax = plt.subplots()
            if hasattr(self,"edges"):
                for i, edge in enumerate(self.edges.T):
                    # ax.plot(self.node_cols[edge],self.node_rows[edge],color="black")
                    if self.node_rows[edge[1]] < self.node_rows[edge[0]]:
                        rotation = 0
                    elif self.node_rows[edge[1]] > self.node_rows[edge[0]]:
                        rotation = 180
                    elif self.node_cols[edge[1]] < self.node_cols[edge[0]]:
                        rotation = -90
                    else:
                        rotation = 90
                    text = f"{self.edge_value[i]:.2f}"
                    if self.edge_type[i]==0:
                        draw_resistor_symbol(ax, x=np.mean(self.node_cols[edge]), y=np.mean(self.node_rows[edge]), rotation=rotation)
                        if self.edge_value[i] >= 1.0:
                            text = f"{self.edge_value[i]:.2f}"
                        else:
                            text = f"{self.edge_value[i]*1e3:.2f}m"
                    elif self.edge_type[i]==1:
                        draw_CC_symbol(ax, x=np.mean(self.node_cols[edge]), y=np.mean(self.node_rows[edge]), rotation=rotation+180)
                    else:
                        draw_diode_symbol(ax, x=np.mean(self.node_cols[edge]), y=np.mean(self.node_rows[edge]), rotation=rotation)
                        text = f"{self.edge_value[i]:.2e}"
                    text_rotation = 0
                    if rotation==90 or rotation==-90:
                        text_rotation = 90
                    ax.text(np.mean(self.node_cols[edge])+np.cos(text_rotation*np.pi/180)*0.2, np.mean(self.node_rows[edge])+np.sin(text_rotation*np.pi/180)*0.2, text, ha="center", va="center", fontsize=8, rotation=90-text_rotation)
            if hasattr(self,"bc"):
                for i in range(self.num_nodes):
                    if not np.isnan(self.bc[i]):
                        if self.bc[i]==0:
                            draw_earth_symbol(ax,x=self.node_cols[i]+0.1-0.025, y=self.node_rows[i]-0.1+0.025,rotation=45)
                        else:
                            draw_pos_terminal_symbol(ax,x=self.node_cols[i], y=self.node_rows[i],color="red")
                            if abs(self.bc[i]) >= 1:
                                text = f"{i}:{self.bc[i]:.3f}"
                            else:
                                text = f"{i}:{self.bc[i]*1e3:.3f}m"
                            ax.text(self.node_cols[i]+0.2, self.node_rows[i]-0.2, text, ha="center", va="center", fontsize=8,color="red")
            if hasattr(self,"voltages"):
                for i in range(self.num_nodes):
                    if abs(self.voltages[i].item()) >= 1:
                        text = f"{i}:{self.voltages[i].item():.3f}"
                        # text = f"{self.voltages[i].item():.3f}"
                    else:
                        text = f"{i}:{self.voltages[i].item()*1e3:.3f}m"
                        # text = f"{self.voltages[i].item():.3f}"
                    ax.text(self.node_cols[i]+0.2, self.node_rows[i]-0.1, text, ha="center", va="center", fontsize=8,color="blue")
            ax.set_xlim(-1,self.cols)
            ax.set_ylim(-1,self.rows)
            plt.show()
    def fill_edges(self,num_edges=None):
        possible_edges = []
        for i in range(self.num_nodes):
            indices = np.where(np.abs(self.node_rows-self.node_rows[i])+np.abs(self.node_cols-self.node_cols[i])<=1)[0]
            for index in indices:
                possible_edges.append([i,index])
        possible_edges = torch.tensor(possible_edges,dtype=torch.long)
        find_ = torch.where(possible_edges[:,0]<possible_edges[:,1])[0]
        possible_edges = possible_edges[find_]
        max_ = possible_edges.shape[0]
        if num_edges is None:
            num_edges = int(max_ * 0.8)
        if num_edges > max_:
            num_edges = max_
        rand_perm = torch.randperm(max_)
        self.edges = possible_edges[rand_perm[:num_edges],:]
        for i in range(self.num_nodes):
            if not torch.any((self.edges[:, 0] == i) | (self.edges[:, 1] == i)):
                is_in_possible = (possible_edges[:, 0] == i) | (possible_edges[:, 1] == i)
                indices = is_in_possible.nonzero(as_tuple=True)[0]
                if indices.numel() > 0:
                    first_edge = possible_edges[indices[0]].unsqueeze(0)
                    self.edges = torch.cat([self.edges, first_edge], dim=0)
        num_edges = self.edges.shape[0]
        polarity = torch.randint(0, 2, (num_edges,))
        find_ = torch.where(polarity == 0)[0]
        self.edges[find_] = self.edges[find_][:, [1, 0]]
        self.edge_type = torch.zeros(num_edges, dtype=torch.long)
        random_number = torch.rand(num_edges)  # uniform [0,1)
        if self.is_linear:
            self.edge_type[random_number < 0.3] = 1
        else:
            self.edge_type[random_number < 0.3] = 1
            self.edge_type[random_number > 0.7] = 2
        for i in range(self.num_nodes):
            if not torch.any(((self.edges[:, 0] == i) | (self.edges[:, 1] == i)) & (self.edge_type == 0)):
                find2_ = torch.where(((self.edges[:,0]==i) | (self.edges[:,1]==i)))[0]
                self.edge_type[find2_[0]] = 0
        self.edge_value = torch.zeros(num_edges)
        find_ = torch.where(self.edge_type==0)[0]
        self.edge_value[find_] = torch.from_numpy(10.0**(np.random.uniform(-3, 2, size=len(find_)))).to(dtype=self.edge_value.dtype)
        find_ = np.where(self.edge_type==1)[0]
        self.edge_value[find_] = torch.from_numpy(np.random.uniform(0.1, 10, size=len(find_))).to(dtype=self.edge_value.dtype)
        find_ = np.where(self.edge_type==2)[0]
        self.edge_value[find_] = torch.from_numpy(10.0**(np.random.uniform(-13, -8, size=len(find_)))).to(dtype=self.edge_value.dtype)
        self.edges = self.edges.T

        data = pyg.data.Data(edge_index=self.edges, num_nodes=self.num_nodes)
        G = to_networkx(data,to_undirected=True)
        self.connected_components = list(nx.connected_components(G))
    def fill_boundary_conditions(self):
        self.bc = np.NaN*np.ones(self.num_nodes)
        for i in range(self.num_nodes):
            find_ = torch.where((self.edges[0,:]==i) | (self.edges[1,:]==i))[0]
            if len(find_)<=1:
                random_number = np.random.rand()
                if random_number < 0.3:
                    self.bc[i] = 0
                else:
                    self.bc[i] = np.random.rand()*10 - 5
        for component in self.connected_components:
            list_ = list(component)
            if not np.any(~np.isnan(self.bc[list_])):
                self.bc[list_[0]] = 0
    def export(self):
        COO_tensor = torch.zeros(2*self.edges.shape[1],7)
        diode_nodes_tensor = torch.zeros(2*self.edges.shape[1],2,dtype=torch.long)
        node_boundary_conditions = torch.zeros(self.num_nodes,3,dtype=torch.float)
        for i, edge in enumerate(self.edges.T):
            COO_tensor[i,0:2] = torch.tensor(edge, dtype=self.edges.dtype)
            # IL, cond, log_I0, n, breakdownV
            edge_feature = torch.zeros(1,5)
            edge_feature[0,3] = 1.0
            # diode neg node, diode pos node
            diode_nodes = torch.tensor([-1,-1], dtype=torch.long)
            if self.edge_type[i]==0:
                edge_feature[0,1] = 1/self.edge_value[i]
            elif self.edge_type[i]==1:
                edge_feature[0,0] = self.edge_value[i]
            elif self.edge_type[i]==2:
                edge_feature[0,2] = -np.log(self.edge_value[i])
                diode_nodes = torch.tensor([edge[1],edge[0]], dtype=torch.long)
            COO_tensor[i,2:] = edge_feature
            diode_nodes_tensor[i,:] = diode_nodes

            COO_tensor[i+self.edges.shape[1],0:2] = torch.tensor(torch.flip(edge, dims=[0]), dtype=self.edges.dtype)
            # IL, cond, log_I0, n, breakdownV
            edge_feature = torch.zeros(1,5)
            edge_feature[0,3] = 1.0
            # diode neg node, diode pos node
            diode_nodes = torch.tensor([-1,-1], dtype=torch.long)
            if self.edge_type[i]==0:
                edge_feature[0,1] = 1/self.edge_value[i]
            elif self.edge_type[i]==1:
                edge_feature[0,0] = -self.edge_value[i]
            elif self.edge_type[i]==2:
                edge_feature[0,2] = np.log(self.edge_value[i])
                diode_nodes = torch.tensor([edge[1], edge[0]], dtype=torch.long)
            COO_tensor[i+self.edges.shape[1],2:] = edge_feature
            diode_nodes_tensor[i+self.edges.shape[1],:] = diode_nodes
        for i in range(self.num_nodes):
            if not np.isnan(self.bc[i]):
                node_boundary_conditions[i,0] = 1
                node_boundary_conditions[i,1] = self.bc[i]
        starting_guess = torch.zeros(self.num_nodes,1)
        edge_index = COO_tensor[:,:2].long().t()
        edge_feature = COO_tensor[:,2:]

        data = pyg.data.Data(
            x=starting_guess,  
            edge_index=edge_index,  
            edge_attr=edge_feature, 
            y=node_boundary_conditions,
            diode_nodes_tensor = diode_nodes_tensor
        )
        
        return data
    
    def solve(self):
        data = self.export()
        cn = CircuitNetwork()
        find_ = torch.where(data.diode_nodes_tensor[:self.edges.shape[1],0]>=0)[0]
        diode_edges = data.diode_nodes_tensor[find_,:]
        rows = torch.where(data.y[:,0] != 1)[0]
        indices = torch.where(data.y[:,0]==1) # pinned voltage boundary condition
        data.x[indices[0],0] = data.y[indices[0],1]
        node_error = cn.forward(data)
        x = data.x
        for _ in range(50):
            J = torch.autograd.functional.jacobian(lambda x_: cn(pyg.data.Data(x=x_, 
                                                                            edge_index=data.edge_index, 
                                                                            edge_attr=data.edge_attr, 
                                                                            diode_nodes_tensor=data.diode_nodes_tensor,
                                                                            y=data.y)), x).squeeze()
            J = J[rows][:, rows]
            Y = -node_error[rows]
            delta_x = torch.zeros_like(x)
            X = torch.linalg.solve(J, Y)
            delta_x[rows] = X
            ratio = 1.0
            if diode_edges.shape[0] > 0:
                delta_diode_V = delta_x[diode_edges[:,1]]-delta_x[diode_edges[:,0]]
                max_diode_V_pos_delta = torch.max(delta_diode_V).item()
                old_diode_V = x[diode_edges[:,1]]-x[diode_edges[:,0]]
                new_diode_V = old_diode_V + delta_diode_V
                max_diode_V_index = torch.argmax(new_diode_V)
                if max_diode_V_pos_delta > self.diode_V_pos_delta_limit:
                    ratio = self.diode_V_pos_delta_limit/max_diode_V_pos_delta
                if new_diode_V[max_diode_V_index] > self.diode_V_hard_limit:
                    ratio = min(ratio,(self.diode_V_hard_limit-old_diode_V[max_diode_V_index])/delta_diode_V[max_diode_V_index])
            x = x + delta_x*ratio
            data.x = x
            node_error = cn.forward(data)
            
            RMS = torch.sqrt(torch.mean(node_error**2))
            if RMS < 1e-4:
                break
        self.voltages = x

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
    COO_tensor, diode_nodes_tensor = coo.generate_COO_tensor()
    return COO_tensor, diode_nodes_tensor
        
class CircuitNetwork(pyg.nn.MessagePassing):
    def forward(self, data, x=None):
        if x is None:
            x = data.x
        
        return self.propagate(data.edge_index, x=(x,x), edge_feature=data.edge_attr, y=data.y, diode_nodes=data.diode_nodes_tensor, ref_x=x)

    def message(self, x_i, x_j, edge_feature, diode_nodes, ref_x):
        # resistor - current is positive from j --> i
        cond = edge_feature[:,1].unsqueeze(1)
        I = cond*(x_j - x_i)
        # current source
        IL = edge_feature[:,0].unsqueeze(1)
        I += IL
        # diode
        find_ = torch.where(diode_nodes[:,0]>=0)[0]
        diode_V_drop = (ref_x[diode_nodes[find_,1],0] - ref_x[diode_nodes[find_,0],0]).unsqueeze(1)
    
        log_I0 = edge_feature[find_,2].unsqueeze(1)
        I0 = -torch.sign(log_I0)*torch.exp(-torch.abs(log_I0))
        n = edge_feature[find_,3].unsqueeze(1)
        breakdownV = edge_feature[find_,4].unsqueeze(1)
        I[find_] -= I0*(torch.exp((diode_V_drop-breakdownV)/(n*0.02568))-1.0)
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
        self.edge_in = MLP(5, hidden_size, hidden_size, 3)
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
        node_feature = torch.cat((self.embed_type(data.x[:,0].long()), data.x[:,1:]), dim=-1)
        node_feature = self.node_in(node_feature)
        edge_feature = self.edge_in(data.edge_attr)
        # stack of GNN layers
        for i in range(self.n_mp_layers):
            node_feature, edge_feature = self.layers[i](node_feature, data.edge_index, edge_feature=edge_feature, diode_nodes=data.diode_nodes_tensor)
        # post-processing
        out = self.node_out(node_feature)

        # indices = torch.where(data.x[:,0]==1)[0] # pinned voltage boundary condition
        # out[indices,0] = data.x[indices,1].float()
        node_error = self.cn.forward(data, out)

        return out, node_error


if __name__ == "__main__":
    cn = CircuitNetwork()
    grid_circuit = GridCircuit()

    with open("grid_circuit.pkl", "wb") as f:
        pickle.dump(grid_circuit, f)

    with open("grid_circuit.pkl", "rb") as f:
        grid_circuit = pickle.load(f)

    grid_circuit.draw()
    
    # data = grid_circuit.export()
    grid_circuit.solve()
    grid_circuit.draw()

    # tandem_model = pickle.load(open(r"C:\Users\arson\Documents\Tandem_Cell_Fit_Tools\best_fit_tandem_model.pkl", 'rb'))
    # tandem_model.cells[0].set_IL(0.0)

    # target_I = 0
    # tandem_model.set_operating_point(I=target_I)
    # V1 = tandem_model.operating_point[0]
    # V3 = V1-tandem_model.subgroups[2].operating_point[0]
    # V2 = V3-tandem_model.subgroups[1].operating_point[0]    
    # insert_PC_currentsource(tandem_model)
    # assign_nodes(tandem_model)
    # tandem_model.draw(display_value=True)
        
    # coo, diode_nodes_tensor = translate_to_COO(tandem_model)
    # edge_index = coo[:,:2].long().t()
    # edge_feature = coo[:,2:]
    # max_node_index = edge_index.max().item()

    # x = torch.tensor([0.0,V1,V2,V3])[:,None]
    # print("kaka1")
    # print(x)
    # # 0th index is the boundary condition:
    # # 0-not a boundary; 1-voltage pinning (Dirichlet boundary condition); 2-net current pinning
    # # 1st index is the voltage to pin (if 0th index is 1)
    # # 2nd index is the net current to pin (if 0th index is 2)
    # node_boundary_conditions = torch.tensor([[1,0,0],[2,0,target_I],[0,0,0],[0,0,0]],dtype=torch.float)
    

    # data = pyg.data.Data(
    #     x=x,  
    #     edge_index=edge_index,  
    #     edge_attr=edge_feature, 
    #     y=node_boundary_conditions,
    #     diode_nodes_tensor = diode_nodes_tensor
    # )

    # node_error = cn.forward(data)
    # rows = torch.where(data.y[:,0] != 1)[0]
    # x = data.x

    # for i in range(5):
    #     J = torch.autograd.functional.jacobian(lambda x_: cn(pyg.data.Data(x=x_, 
    #                                                                        edge_index=data.edge_index, 
    #                                                                        edge_attr=data.edge_attr, 
    #                                                                        diode_nodes_tensor=data.diode_nodes_tensor,
    #                                                                        y=data.y)), x).squeeze()
    #     J = J[rows][:, rows]

    #     node_error = node_error[rows]
    #     Y = -node_error

    #     X = torch.linalg.solve(J, Y)

    #     x[rows] = x[rows] + X

    #     data.x = x

    #     node_error = cn.forward(data)
    
    # print("kaka")
    # print(x)
    # print(node_error)
