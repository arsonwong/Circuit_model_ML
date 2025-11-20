from Circuit_Model_ML.circuit_network import *
from utilities import get_data_generation_settings

if __name__ == "__main__":

    zeroth_iteration, is_linear, has_current_source, acceptable_initial_cond_num = get_data_generation_settings()
    grid_circuit = GridCircuit(rows=4,cols=4,node_density=0.8,edge_density=0.8,is_linear=is_linear,has_current_source=has_current_source) # makes a random grid circuit each time
    _, aux = grid_circuit.solve(convergence_RMS=1e-8, acceptable_initial_cond_num=acceptable_initial_cond_num)
    grid_circuit.draw()
    