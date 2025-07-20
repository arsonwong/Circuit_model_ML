import torch
import torch_geometric as pyg
import torch_scatter
import numpy as np
from matplotlib import pyplot as plt
from torch_geometric.utils import to_networkx
import networkx as nx
from .utilities import *
    
class GridCircuit():
    diode_V_hard_limit = 0.8
    diode_V_pos_delta_limit = 0.5
    def __init__(self,rows=4,cols=4,is_linear=False,has_current_source=True):
        self.rows = rows
        self.cols = cols
        self.is_linear = is_linear
        self.has_current_source = has_current_source
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
            if self.has_current_source:
                self.edge_type[random_number < 0.3] = 1
        else:
            if self.has_current_source:
                self.edge_type[random_number < 0.3] = 1
                self.edge_type[random_number > 0.7] = 2
            else:
                self.edge_type[random_number > 0.6] = 2
            
        for i in range(self.num_nodes):
            if not torch.any(((self.edges[:, 0] == i) | (self.edges[:, 1] == i)) & (self.edge_type == 0)):
                find2_ = torch.where(((self.edges[:,0]==i) | (self.edges[:,1]==i)))[0]
                self.edge_type[find2_[0]] = 0
        self.edge_value = torch.zeros(num_edges, dtype=torch.double)
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
        COO_tensor = torch.zeros(2*self.edges.shape[1],7,dtype=torch.double)
        diode_nodes_tensor = torch.zeros(2*self.edges.shape[1],2,dtype=torch.long)
        node_boundary_conditions = torch.zeros(self.num_nodes,3,dtype=torch.double)
        for i, edge in enumerate(self.edges.T):
            COO_tensor[i,0:2] = edge.clone().detach()
            # IL, cond, log_I0, n, breakdownV
            edge_feature = torch.zeros(1,5,dtype=torch.double)
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

            COO_tensor[i+self.edges.shape[1],0:2] = torch.flip(edge, dims=[0]).clone().detach()
            # IL, cond, log_I0, n, breakdownV
            edge_feature = torch.zeros(1,5,dtype=torch.double)
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
        starting_guess = torch.zeros(self.num_nodes,1,dtype=torch.double)
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
    def solve(self, convergence_RMS=1e-8, suppress_warning=False):
        data = self.export()
        success, x, RMS = solve_circuit(data, convergence_RMS=convergence_RMS, suppress_warning=suppress_warning)
        if x is not None:
            self.voltages = x
        return success, RMS
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

def solve_circuit(data,diode_V_pos_delta_limit=0.5,diode_V_hard_limit=0.8,convergence_RMS=1e-8, suppress_warning=False):
    cn = CircuitNetwork()
    find_ = torch.where(data.diode_nodes_tensor[:data.edge_index.shape[1] // 2,0]>=0)[0]
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
        try:
            X = torch.linalg.solve(J, Y)
        except Exception as e:
            if not suppress_warning:
                print(f"Linear solver error: {e}")
            return False, None, None
        delta_x[rows] = X
        ratio = 1.0
        if diode_edges.shape[0] > 0:
            delta_diode_V = delta_x[diode_edges[:,1]]-delta_x[diode_edges[:,0]]
            max_diode_V_pos_delta = torch.max(delta_diode_V).item()
            old_diode_V = x[diode_edges[:,1]]-x[diode_edges[:,0]]
            new_diode_V = old_diode_V + delta_diode_V
            max_diode_V_index = torch.argmax(new_diode_V)
            if max_diode_V_pos_delta > diode_V_pos_delta_limit:
                ratio = diode_V_pos_delta_limit/max_diode_V_pos_delta
            if new_diode_V[max_diode_V_index] > diode_V_hard_limit:
                ratio = min(ratio,(diode_V_hard_limit-old_diode_V[max_diode_V_index])/delta_diode_V[max_diode_V_index])
        x = x + delta_x*ratio
        data.x = x
        node_error = cn.forward(data)
        
        RMS = torch.sqrt(torch.mean(node_error**2)).item()
        if RMS < convergence_RMS:
            break
    if RMS < convergence_RMS:
        return True, x, RMS
    if not suppress_warning:
        print("Non convergence: RMS = ", RMS)
    return False, x, RMS
        
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

if __name__ == "__main__":
    grid_circuit = GridCircuit()
    grid_circuit.draw()
    _, RMS = grid_circuit.solve(convergence_RMS=1e-8)
    grid_circuit.draw()
    print(RMS)
    