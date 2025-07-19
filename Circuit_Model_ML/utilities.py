import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as patches

from PV_Circuit_Model.circuit_model import *
from PV_Circuit_Model.cell import *
from PV_Circuit_Model.multi_junction_cell import *
import torch

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


def rotate_points(xy_pairs, origin, angle_deg):
    angle_rad = np.deg2rad(angle_deg)
    rot_matrix = np.array([
        [np.cos(angle_rad), -np.sin(angle_rad)],
        [np.sin(angle_rad),  np.cos(angle_rad)]
    ])
    rotated = []
    ox, oy = origin
    for x, y in xy_pairs:
        vec = np.array([x - ox, y - oy])
        rx, ry = rot_matrix @ vec + np.array([ox, oy])
        rotated.append((rx, ry))
    return rotated

def draw_diode_symbol(ax, x=0, y=0, color="black", up_or_down="down", is_LED=False, rotation=0):
    origin = (x, y)
    dir = 1 if up_or_down == "down" else -1

    # Diode circle for LED
    if is_LED:
        circle = patches.Circle(origin, 0.17, edgecolor=color, facecolor='white', linewidth=1.5, fill=False)
        ax.add_patch(circle)

    # Diode triangle arrowhead
    arrow_start = (x, y + 0.075 * dir)
    arrow_end = (x, y + 0.074 * dir - 0.001 * dir)  # Very short shaft
    [(x0, y0), (x1, y1)] = rotate_points([arrow_start, arrow_end], origin, rotation)
    ax.arrow(x0, y0, x1 - x0, y1 - y0, head_width=0.15, head_length=0.15, fc=color, ec=color)

    # Diode bar
    bar_pts = rotate_points([(x - 0.075, y - 0.08 * dir), (x + 0.075, y - 0.08 * dir)], origin, rotation)
    ax.add_line(plt.Line2D([bar_pts[0][0], bar_pts[1][0]], [bar_pts[0][1], bar_pts[1][1]], color=color, linewidth=2))

    # LED rays
    if is_LED:
        ray1 = rotate_points([(x - 0.05, y - 0.05 * dir), (x - 0.2, y - 0.2 * dir)], origin, rotation)
        ray2 = rotate_points([(x - 0.075, y + 0.025 * dir), (x - 0.225, y - 0.125 * dir)], origin, rotation)
        ax.arrow(*ray1[0], ray1[1][0] - ray1[0][0], ray1[1][1] - ray1[0][1],
                 head_width=0.05, head_length=0.05, fc='orange', ec='orange')
        ax.arrow(*ray2[0], ray2[1][0] - ray2[0][0], ray2[1][1] - ray2[0][1],
                 head_width=0.05, head_length=0.05, fc='orange', ec='orange')

    # Terminals
    term_top = rotate_points([(x, y + 0.08), (x, y + 0.5)], origin, rotation)
    term_bot = rotate_points([(x, y - 0.08), (x, y - 0.5)], origin, rotation)
    ax.add_line(plt.Line2D([term_top[0][0], term_top[1][0]], [term_top[0][1], term_top[1][1]], color=color, linewidth=1.5))
    ax.add_line(plt.Line2D([term_bot[0][0], term_bot[1][0]], [term_bot[0][1], term_bot[1][1]], color=color, linewidth=1.5))

def draw_CC_symbol(ax, x=0, y=0, color="black", rotation=0):
    origin = (x, y)

    # Draw rotated circle
    circle = patches.Circle((x, y), 0.17, edgecolor=color, facecolor="white", linewidth=2)
    ax.add_patch(circle)

    # Arrow inside circle (from lower to upper)
    arrow_start = (x, y - 0.12)
    arrow_end = (x, y + 0.02)
    [(x0, y0), (x1, y1)] = rotate_points([arrow_start, arrow_end], origin, rotation)
    ax.arrow(x0, y0, x1 - x0, y1 - y0, head_width=0.1, head_length=0.1, width=0.01, fc=color, ec=color)

    # Vertical terminals (above and below the circle)
    line1 = rotate_points([(x, y + 0.18), (x, y + 0.5)], origin, rotation)
    line2 = rotate_points([(x, y - 0.18), (x, y - 0.5)], origin, rotation)
    ax.add_line(plt.Line2D([line1[0][0], line1[1][0]], [line1[0][1], line1[1][1]], color=color, linewidth=1.5))
    ax.add_line(plt.Line2D([line2[0][0], line2[1][0]], [line2[0][1], line2[1][1]], color=color, linewidth=1.5))

def draw_resistor_symbol(ax, x=0, y=0, color="black", rotation=0):
    dx = 0.075
    dy = 0.02
    ystart = y + 0.15
    origin = (x, y)

    # Top and bottom vertical leads
    segments = [
        [(x, y+0.15), (x, y+0.5)],
        [(x, y-0.09), (x, y-0.5)],
    ]

    # Zigzag segments
    for _ in range(3):
        segments += [
            [(x, ystart), (x+dx, ystart-dy)],
            [(x+dx, ystart-dy), (x-dx, ystart-3*dy)],
            [(x-dx, ystart-3*dy), (x, ystart-4*dy)]
        ]
        ystart -= 4 * dy

    for (x0, y0), (x1, y1) in segments:
        [(x0r, y0r), (x1r, y1r)] = rotate_points([(x0, y0), (x1, y1)], origin, rotation)
        ax.add_line(plt.Line2D([x0r, x1r], [y0r, y1r], color=color, linewidth=1.5))

def draw_earth_symbol(ax, x=0, y=0, color="black", rotation=0):
    origin = (x, y)
    segments = []
    # Vertical line
    segments.append([(x, y - 0.05), (x, y + 0.1)])
    # Horizontal lines
    for i in range(3):
        x1 = x - 0.03 * (i + 1)
        x2 = x + 0.03 * (i + 1)
        y_level = y - 3*0.05 + 0.05 * i
        segments.append([(x1, y_level), (x2, y_level)])
    for (x0, y0), (x1, y1) in segments:
        [(x0r, y0r), (x1r, y1r)] = rotate_points([(x0, y0), (x1, y1)], origin, rotation)
        ax.add_line(plt.Line2D([x0r, x1r], [y0r, y1r], color=color, linewidth=2))

def draw_pos_terminal_symbol(ax, x=0, y=0, color="black"):
    circle = patches.Circle((x, y), 0.04, edgecolor=color,facecolor="white",linewidth=2, fill=True)
    ax.add_patch(circle)


