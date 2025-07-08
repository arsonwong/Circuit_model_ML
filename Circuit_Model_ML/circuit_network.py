'''

You can implement Newton solvers yourself using PyTorch (which gives you autograd support for Jacobians), 
and use PyG to build your graph-based representations or neural networks.

Newton solver:
Kirchoff's law - M is a matrix of dI(into node i)/dVj, thus a Jacobian
matrix equation goes like: dI/dV * dV = desired dI to reach boundary cond
except at rows where there's a boundary condition

So to use the autograd we need to progagte the following
first define element I(V), like resistor I = v/R
rewrite as element I(V1,V2) where the v1 and v2 are its two ends
then use pyG to define the message passing such that you form the I(into node) as funciton of Vs
and now you can use autograd to get the Jacobian

define CSR(index1,index2,element)
through this, we latch the element's V1=V(index1) and V2=V(index2)
we also aggregate the element's I=I(index1) and I=-I(index2)
it should be element V(index1)-->V1-->I-->I(index1) and -I(index2)
formally it has the strucure of node features (V) --> encode --> process --> decode --> node features (I)

then we can get the Jacobian dI/dV


'''

from PV_Circuit_Model.circuit_model import *
from PV_Circuit_Model.cell import *
from PV_Circuit_Model.multi_junction_cell import *
import torch
import torch_geometric as pyg
import torch_scatter
import pickle

tandem_model = pickle.load(open(r"C:\Users\arson\Documents\Tandem_Cell_Fit_Tools\best_fit_tandem_model.pkl", 'rb'))
tandem_model.set_operating_point(V = 3)
# print(tandem_model.operating_point)
V1 = tandem_model.operating_point[0]
# tandem_model.plot(fourth_quadrant=False)
# tandem_model.show()
# for element in tandem_model.subgroups:
#     print(element.operating_point)
#     element.plot(fourth_quadrant=False)
#     element.show()

V3 = V1-tandem_model.subgroups[2].operating_point[0]
V2 = V3-tandem_model.subgroups[1].operating_point[0]

# need COO (index1,index2,element)
class COO():
    def __init__(self):
        self.COO_list = []
    def insert(self,index1,index2,element:CircuitElement):
        self.COO_list.append([index1,index2,element])
    @staticmethod
    def generate_edge_feature(element:CircuitElement):
        area = 1.0
        if element.parent is not None and isinstance(element.parent,Cell):
            area = element.parent.area
        # IL, cond, I0, n, breakdownV, diode polarity
        edge_feature = torch.zeros(1,6)
        edge_feature[0,3] = 1.0
        edge_feature[0,5] = 1.0
        if isinstance(element,CurrentSource):
            edge_feature[0,0] = element.IL*area
        elif isinstance(element,Resistor):
            edge_feature[0,1] = element.cond*area
        elif isinstance(element,Diode):
            edge_feature[0,2] = element.I0*area
            edge_feature[0,3] = element.n
            edge_feature[0,4] = element.V_shift
            if isinstance(element,ReverseDiode):
                edge_feature[0,5] = -1.0

        return edge_feature
    def generate_COO_tensor(self):
        COO_tensor = torch.zeros(2*len(self.COO_list),9)
        for i, _ in enumerate(self.COO_list):
            COO_tensor[i,0:2] = torch.tensor(self.COO_list[i][0:2], dtype=COO_tensor.dtype)
            COO_tensor[i,2] = 1.0 # 1.0 if COO_tensor[i,0:2] is neg_node and pos_node; -1.0 if it is pos_node and neg_node
            COO_tensor[i,3:] = self.generate_edge_feature(self.COO_list[i][-1])
            COO_tensor[i+len(self.COO_list),0:2] = torch.tensor(self.COO_list[i][0:2][::-1], dtype=COO_tensor.dtype)
            COO_tensor[i+len(self.COO_list),2] = -1.0 # 1.0 if COO_tensor[i,0:2] is neg_node and pos_node; -1.0 if it is pos_node and neg_node
            COO_tensor[i+len(self.COO_list),3:] = self.generate_edge_feature(self.COO_list[i][-1])
        return COO_tensor
    
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

assign_nodes(tandem_model)
tandem_model.draw(display_value=True)

def translate_to_COO(circuit_group):
    coo = COO()
    assign_nodes(circuit_group)
    circuit_elements = circuit_group.findElementType(CircuitElement)            
    for element in circuit_elements:
        coo.insert(element.neg_node,element.pos_node,element)
    return coo.generate_COO_tensor()
        
class CircuitNetwork(pyg.nn.MessagePassing):
    def forward(self, x, edge_index, edge_feature):
        return self.propagate(edge_index, x=(x, x), edge_feature=edge_feature)

    def message(self, x_i, x_j, edge_feature):
        V = x_i - x_j
        polarity = edge_feature[:,0].unsqueeze(1)
        V *= polarity
        # IL, cond, I0, n, breakdownV
        IL = edge_feature[:,1].unsqueeze(1)
        cond = edge_feature[:,2].unsqueeze(1)
        I0 = edge_feature[:,3].unsqueeze(1)
        n = edge_feature[:,4].unsqueeze(1)
        breakdownV = edge_feature[:,5].unsqueeze(1)
        diode_polarity = edge_feature[:,6].unsqueeze(1)
        # I = -polarity * cond*V
        # I = polarity * IL
        # I = -polarity * (diode_polarity * I0*(torch.exp((V*diode_polarity-breakdownV)/(n*0.02568))-0.5*(1+diode_polarity)))
        
        I = polarity * (IL - cond*V - diode_polarity * I0*(torch.exp((V*diode_polarity-breakdownV)/(n*0.02568))-0.5*(1+diode_polarity)))
        
        # I = polarity * (IL - cond*V)
        #I = polarity * (IL + cond*V + diode_polarity * I0*(torch.exp((V*diode_polarity-breakdownV)/(n*0.02568))-1.0))
        return I
    
    def aggregate(self, inputs, index, dim_size=None):
        return torch_scatter.scatter(inputs, index, dim=self.node_dim, dim_size=dim_size, reduce="sum")
    
coo = translate_to_COO(tandem_model)
edge_index = coo[:,:2].long().t()
edge_feature = coo[:,2:]
max_node_index = edge_index.max().item()
# x = torch.zeros(max_node_index+1,1)
# x = torch.tensor([0.0,-10.7,-10.7,-10.7])[:,None]
x = torch.tensor([0.0,V1,V2,V3])[:,None]
cn = CircuitNetwork()
result = cn.forward(x, edge_index, edge_feature)

print(result)