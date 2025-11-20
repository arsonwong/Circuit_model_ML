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
    def __init__(self,rows=4,cols=4,node_density=0.8,edge_density=0.8,is_linear=False,has_current_source=True):
        self.rows = rows
        self.cols = cols
        self.edge_density = edge_density
        self.node_density = node_density
        self.is_linear = is_linear
        self.has_current_source = has_current_source
        self.fill_nodes()
        self.fill_edges()
        self.fill_boundary_conditions()
    def fill_nodes(self,num_nodes=None):
        max_ = self.rows*self.cols
        if num_nodes is None:
            num_nodes = int(max_ * self.node_density)
        rand_perm = np.random.permutation(max_)
        self.num_nodes = num_nodes
        self.node_pos = rand_perm[:num_nodes]
        self.node_rows = np.floor(self.node_pos / self.cols)
        self.node_cols = self.node_pos - self.node_rows*self.cols
        # take out nodes that end up on an island
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
        # single directed edges from lower index node to higher index node
        possible_edges = []
        for i in range(self.num_nodes):
            indices = np.where((np.arange(self.num_nodes)>i) & (np.abs(self.node_rows-self.node_rows[i])+np.abs(self.node_cols-self.node_cols[i])<=1))[0]
            possible_edges.extend([[i, index] for index in indices])
        possible_edges = torch.tensor(possible_edges,dtype=torch.long)
        max_ = possible_edges.shape[0]
        if num_edges is None:
            num_edges = int(max_ * self.edge_density)
        rand_perm = torch.randperm(max_)
        self.edges = possible_edges[rand_perm[:num_edges],:]
        # if any node became totally disconnected, then connect back to a neighbour
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
        # single directed edges from one index to another
        self.edges[find_] = self.edges[find_][:, [1, 0]]
        self.edge_type = torch.zeros(num_edges, dtype=torch.long)
        random_number = torch.rand(num_edges)  # uniform [0,1)
        if self.is_linear:
            if self.has_current_source:
                find_ = np.where(random_number < 0.3)[0]
                if len(find_) > 0:
                    self.edge_type[find_[0]] = 1
        else:
            if self.has_current_source:
                find_ = np.where(random_number < 0.3)[0]
                if len(find_) > 0:
                    self.edge_type[find_[0]] = 1
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
                self.bc[list_[-1]] = np.random.rand()*10 - 5
    def export(self):
        COO_tensor = torch.zeros(2*self.edges.shape[1],6,dtype=torch.double)
        node_boundary_conditions = torch.zeros(self.num_nodes,2,dtype=torch.double)
        for i, edge in enumerate(self.edges.T):
            # edge[0] to edge[1]
            COO_tensor[i,0:2] = edge.clone().detach()
            # IL, cond, log_I0, diode polarity 
            # diode polarity --> if edge is i,j, and diode points from i to j, then diode polarity is 1
            # otherwise it is -1
            edge_feature = torch.zeros(1,4,dtype=torch.double)
            # diode neg node, diode pos node
            if self.edge_type[i]==0:
                edge_feature[0,1] = 1/self.edge_value[i]
            elif self.edge_type[i]==1:
                edge_feature[0,0] = self.edge_value[i]
            elif self.edge_type[i]==2:
                edge_feature[0,2] = np.log(self.edge_value[i])
                # diode polarity --> if edge is i,j, and diode points from i to j, then diode polarity is 1
                # otherwise it is -1
                edge_feature[0,3] = 1
            COO_tensor[i,2:] = edge_feature

            # edge[1] to edge[0]
            j = i+self.edges.shape[1]
            COO_tensor[j,0:2] = torch.flip(edge, dims=[0]).clone().detach()
            COO_tensor[j,2:] = edge_feature
            if self.edge_type[i]==0:
                pass 
            elif self.edge_type[i]==1:
                COO_tensor[j,2] *= -1 # flip the sign of the current
            elif self.edge_type[i]==2:
                COO_tensor[j,5] *= -1 # flip the sign of the diode polarity

        for i in range(self.num_nodes):
            if not np.isnan(self.bc[i]):
                node_boundary_conditions[i,0] = 1
                node_boundary_conditions[i,1] = self.bc[i]
        starting_guess = torch.zeros(self.num_nodes,1,dtype=torch.double)
        edge_index = COO_tensor[:,:2].long().t()
        edge_feature = COO_tensor[:,2:]

        data = pyg.data.Data(
            x=starting_guess,  #  nodes x 1
            edge_index=edge_index,  # edges x 2
            edge_attr=edge_feature, # edges x 4
            y=node_boundary_conditions, # nodes x 2
        )
        
        return data
    def solve(self, convergence_RMS=1e-8, acceptable_initial_cond_num=None, suppress_warning=False):
        data = self.export()
        success, x, aux = solve_circuit(data, convergence_RMS=convergence_RMS, acceptable_initial_cond_num=acceptable_initial_cond_num, suppress_warning=suppress_warning)
        if x is not None:
            self.voltages = x
        return success, aux
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
            
                for i in range(self.num_nodes):
                    if hasattr(self,"voltages"):
                        if abs(self.voltages[i].item()) >= 1:
                            text = f"{i}:{self.voltages[i].item():.3f}"
                            # text = f"{self.voltages[i].item():.3f}"
                        else:
                            text = f"{i}:{self.voltages[i].item()*1e3:.3f}m"
                            # text = f"{self.voltages[i].item():.3f}"
                    else:
                        text = f"{i}"
                    ax.text(self.node_cols[i]+0.2, self.node_rows[i]-0.1, text, ha="center", va="center", fontsize=8,color="blue")
            ax.set_xlim(-1,self.cols)
            ax.set_ylim(-1,self.rows)
            plt.show()

def solve_circuit(data,diode_V_pos_delta_limit=0.5,diode_V_hard_limit=0.8,convergence_RMS=1e-8, 
                  suppress_warning=False,acceptable_initial_cond_num=None,return_num_iterations=False,max_log_diode_I=None):
    cn = CircuitNetwork(max_log_diode_I=max_log_diode_I)
    rows = torch.where(data.y[:,0] != 1)[0]
    indices = torch.where(data.y[:,0]==1) # pinned voltage boundary condition
    data.x[indices[0],0] = data.y[indices[0],1]
    node_error = cn.forward(data)
    RMS = torch.sqrt(torch.mean(node_error**2)).item()
    x = data.x
    record = []
    diode_turned_on = False

    polarity = data.edge_attr[:,3]
    find_ = torch.where(polarity != 0)[0] # diode connections have polarity of 1 or -1
    polarity_ = polarity[find_].unsqueeze(1)
    diode_edges = data.edge_index[:,find_].T

    for i in range(50):
        J = torch.autograd.functional.jacobian(lambda x_: cn(pyg.data.Data(x=x_, 
                                                                        edge_index=data.edge_index, 
                                                                        edge_attr=data.edge_attr, 
                                                                        y=data.y)), x).squeeze()
        J = J[rows][:, rows]
        if i==0 and acceptable_initial_cond_num is not None:
            if torch.linalg.cond(J) > acceptable_initial_cond_num:
                return False, None, None

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
            delta_diode_V = polarity_*(delta_x[diode_edges[:,1]]-delta_x[diode_edges[:,0]])
            max_diode_V_pos_delta = torch.max(delta_diode_V).item()
            old_diode_V = polarity_*(x[diode_edges[:,1]]-x[diode_edges[:,0]])
            new_diode_V = old_diode_V + delta_diode_V
            max_diode_V_index = torch.argmax(new_diode_V)
            if max_diode_V_pos_delta > diode_V_pos_delta_limit:
                ratio = diode_V_pos_delta_limit/max_diode_V_pos_delta
            if new_diode_V[max_diode_V_index] > diode_V_hard_limit:
                diode_turned_on = True
                ratio_ = (diode_V_hard_limit-old_diode_V[max_diode_V_index])/delta_diode_V[max_diode_V_index]
                ratio_ = ratio_.item()
                ratio = min(ratio,ratio_)
        delta_x_altered = delta_x*ratio
        record.append({"x": x.clone().detach(), "delta_x": delta_x.clone().detach(), "RMS": RMS, 
                       "delta_x_altered": delta_x_altered.clone().detach()})
        x = x + delta_x*ratio
        data.x = x
        node_error = cn.forward(data)
        
        RMS = torch.sqrt(torch.mean(node_error**2)).item()
        if RMS < convergence_RMS:
            break
    aux = {'RMS': RMS, 'record': record, 'diode_turned_on': diode_turned_on}
    if RMS < convergence_RMS:
        if return_num_iterations:
            return True, x, i
        return True, x, aux
    if not suppress_warning:
        print("Non convergence: RMS = ", RMS)
    return False, x, aux
        
class CircuitNetwork(pyg.nn.MessagePassing):
    def __init__(self,max_log_diode_I = None):
        super().__init__()
        self.max_log_diode_I = max_log_diode_I

    def forward(self, data, x=None):
        if x is None:
            x = data.x
        
        return self.propagate(data.edge_index, x=(x,x), edge_feature=data.edge_attr, y=data.y)

    def message(self, x_i, x_j, edge_feature):
        # current is positive from j --> i, i.e. going into left node
        # resistor
        cond = edge_feature[:,1].unsqueeze(1)
        I = cond*(x_j - x_i)

        # current source
        IL = edge_feature[:,0].unsqueeze(1)
        I += IL
        # diode.  If polarity = 1, means diode points from i to j, diode_V_drop = x_i - x_j
        # I = -I0*(exp(diode_V_drop)-1) according to convention
        # If polarity = -1, means diode points from j to i, diode_V_drop = -(x_i - x_j)
        # I = +I0*(exp(diode_V_drop)-1) according to convention
        # so, I = -polarity * (exp(polarity * (x_i - x_j))-1)
        polarity = edge_feature[:,3].unsqueeze(1)
        find_ = torch.where(polarity != 0)[0] # diode connections have polarity of 1 or -1

        diode_V_drop = polarity[find_]*(x_i[find_] - x_j[find_])
        log_I0 = edge_feature[find_,2].unsqueeze(1)
        polarity_ = edge_feature[find_,3].unsqueeze(1)
        I0 = torch.exp(log_I0)
        
        if self.max_log_diode_I is not None:
            max_log_diode_I = torch.tensor(self.max_log_diode_I, device=edge_feature.device)
            max_diode_I = torch.exp(max_log_diode_I)
            max_V = (0.02568)*(max_log_diode_I + torch.abs(log_I0))
            find2_ = torch.where(diode_V_drop <= max_V)[0]
            find3_ = torch.where(diode_V_drop > max_V)[0]
            I[find_[find2_]] -= polarity_[find2_]*I0[find2_]*(torch.exp(diode_V_drop[find2_]/0.02568)-1.0)
            I[find_[find3_]] -= polarity_[find3_]*(max_diode_I + (diode_V_drop[find3_]-max_V[find3_])*max_diode_I/0.02568)
        else:
            I[find_] -= polarity_*I0*(torch.exp(diode_V_drop/0.02568)-1.0)
        return I
    
    def aggregate(self, inputs, index, dim_size=None, y=None):
        net_I = torch_scatter.scatter(inputs, index, dim=self.node_dim, dim_size=dim_size, reduce="sum")
        node_error = net_I
        if y is not None:
            indices = torch.where(y[:,0]==1) # pinned voltage boundary condition
            node_error[indices] = 0.0 # just assume the voltage pinned nodes are set at the desired voltage, we don't measure voltage errors
            # indices = torch.where(y[:,0]==2) # pinned current boundary condition
            # node_error[indices] = net_I[indices] - y[indices[0],2].unsqueeze(1) 
        return node_error

if __name__ == "__main__":
    grid_circuit = GridCircuit()
    grid_circuit.draw()
    _, RMS = grid_circuit.solve(convergence_RMS=1e-8)
    grid_circuit.draw()
    print(RMS)
    