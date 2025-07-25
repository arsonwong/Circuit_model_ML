from PV_Circuit_Model.circuit_model import *
from PV_Circuit_Model.cell import *
from PV_Circuit_Model.multi_junction_cell import *
import torch
from circuit_network import *
import pickle

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

def solve_example_tandem_cell():
    cn = CircuitNetwork()
    tandem_model = pickle.load(open(r"C:\Users\arson\Documents\Tandem_Cell_Fit_Tools\best_fit_tandem_model.pkl", 'rb'))
    tandem_model.cells[0].set_IL(0.0)

    target_I = 0
    tandem_model.set_operating_point(I=target_I)
    V1 = tandem_model.operating_point[0]
    V3 = V1-tandem_model.subgroups[2].operating_point[0]
    V2 = V3-tandem_model.subgroups[1].operating_point[0]    
    insert_PC_currentsource(tandem_model)
    assign_nodes(tandem_model)
    tandem_model.draw(display_value=True)
        
    coo, diode_nodes_tensor = translate_to_COO(tandem_model)
    edge_index = coo[:,:2].long().t()
    edge_feature = coo[:,2:]
    max_node_index = edge_index.max().item()

    x = torch.tensor([0.0,V1,V2,V3],dtype=torch.double)[:,None]
    x = torch.tensor([0.0,0,0,0],dtype=torch.double)[:,None]

    print(x)
    # 0th index is the boundary condition:
    # 0-not a boundary; 1-voltage pinning (Dirichlet boundary condition); 2-net current pinning
    # 1st index is the voltage to pin (if 0th index is 1)
    # 2nd index is the net current to pin (if 0th index is 2)
    node_boundary_conditions = torch.tensor([[1,0,0],[2,0,target_I],[0,0,0],[0,0,0]],dtype=torch.double)
    
    data = pyg.data.Data(
        x=x,  
        edge_index=edge_index,  
        edge_attr=edge_feature, 
        y=node_boundary_conditions,
        diode_nodes_tensor = diode_nodes_tensor
    )
    node_error = cn.forward(data)
    print(node_error)

    _, x, aux = solve_circuit(data,diode_V_pos_delta_limit=0.2,diode_V_hard_limit=1.5, convergence_RMS=1e-100)

    print(x)
    print(aux["RMS"])

if __name__ == "__main__":
    solve_example_tandem_cell()