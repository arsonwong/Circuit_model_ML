from Circuit_Model_ML.circuit_network import *
import yaml

def get_data_generation_settings():
    with open("data_generation_settings.yaml", "r") as f:
        settings = yaml.safe_load(f)
        zeroth_iteration = settings["zeroth_iteration"]
        is_linear = settings["is_linear"]
        has_current_source = settings["has_current_source"]
        acceptable_initial_cond_num = settings["acceptable_initial_cond_num"]
    return zeroth_iteration, is_linear, has_current_source, acceptable_initial_cond_num

if __name__ == "__main__":

    zeroth_iteration, is_linear, has_current_source, acceptable_initial_cond_num = get_data_generation_settings()
    grid_circuit = GridCircuit(rows=4,cols=4,is_linear=is_linear,has_current_source=has_current_source) # makes a random grid circuit each time
    _, aux = grid_circuit.solve(convergence_RMS=1e-8, acceptable_initial_cond_num=acceptable_initial_cond_num)
    grid_circuit.draw()
    print(aux["RMS"])
    