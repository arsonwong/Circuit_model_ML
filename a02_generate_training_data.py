from Circuit_Model_ML.circuit_network import *
from tqdm import tqdm 
import pickle
import os
import torch
import random
import numpy as np
import re
from a01_example_circuit_network import get_data_generation_settings

def get_data_folder(is_linear, has_current_source, zeroth_iteration):
    if is_linear:
        if has_current_source:
            folder = "data/linear with current source" 
        else:
            folder = "data/linear with no current source"
    else: 
        if has_current_source:
            folder = "data/nonlinear with current source" 
        else:
            folder = "data/nonlinear with no current source"
    if zeroth_iteration:
        folder += " zeroth iteration"
    return folder

def generate_data(zeroth_iteration = True, is_linear=False, has_current_source=True, acceptable_initial_cond_num=1e3, overwrite=False):
    folder = get_data_folder(is_linear, has_current_source, zeroth_iteration)
    data_num = {"train": 100000, "val": 10000, "val2": 1000}

    for iter in ["train","val", "val2"]:
        filepath = folder+"/"+iter
        os.makedirs(filepath, exist_ok=True)

        if overwrite:
            for filename in tqdm(os.listdir(filepath),desc="Erasing contents: "):
                file_path = os.path.join(filepath, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)

        pbar = tqdm(total=data_num[iter],desc="Generating " + iter + " data")
        max_number = -1
        pattern = re.compile(r"sample_(\d+)\.pkl")
        for filename in os.listdir(filepath):
            match = pattern.match(filename)
            if match:
                number = int(match.group(1))
                max_number = max(max_number, number)
        if max_number >= 0 and overwrite==False:
            pbar.update(max_number+1)
        
        outputs = []
        while pbar.n < data_num[iter]:
            grid_circuit = GridCircuit(rows=4,cols=4,is_linear=is_linear,has_current_source=has_current_source) # makes a random grid circuit each time
            find_ = np.where(grid_circuit.edge_type==1)[0] # looks for current source
            actually_has_current_source = len(find_)>0
            # encourages more circuits with current source
            if has_current_source and (not actually_has_current_source) and np.random.rand()<0.7:
                continue
            find_ = np.where(grid_circuit.edge_type==2)[0] # looks for diodes
            actually_has_diode = len(find_)>0
            # encourages more circuits with diodes
            if (not is_linear) and (not actually_has_diode) and np.random.rand()<0.7:
                continue
            data = grid_circuit.export()
            success, aux = grid_circuit.solve(convergence_RMS=1e-8,acceptable_initial_cond_num=acceptable_initial_cond_num,suppress_warning=True)
            if success:
                variance = torch.var(grid_circuit.voltages)
                if variance > 0:
                    filename = f"sample_{pbar.n}.pkl"
                    diode_turned_on = aux["diode_turned_on"]
                    record = aux["record"]  
                    if (not is_linear) or (diode_turned_on and len(record)>1): 
                        if zeroth_iteration:
                            random_iteration = record[0]         
                        else:
                            random_iteration = random.choice(record)
                        outputs.append({"data": data, 
                                    "answer": grid_circuit.voltages, 
                                    "RMS_record": random_iteration["RMS"], 
                                    "x_record": torch.tensor(random_iteration["x"]), 
                                    "delta_x_record": torch.tensor(random_iteration["delta_x"]),
                                    "delta_x_altered_record": torch.tensor(random_iteration["delta_x_altered"]),
                                    "diode_turned_on": diode_turned_on,
                                    "actually_has_current_source": actually_has_current_source,
                                    "actually_has_diode": actually_has_diode
                                    })
                        if len(outputs)==100:
                            with open(os.path.join(filepath,filename), "wb") as f:
                                pickle.dump(outputs, f)
                            outputs = []
                        pbar.update(1)

if __name__ == "__main__":
    zeroth_iteration, is_linear, has_current_source, acceptable_initial_cond_num = get_data_generation_settings()
    generate_data(zeroth_iteration, is_linear, has_current_source, acceptable_initial_cond_num, overwrite=True)