from Circuit_Model_ML.circuit_network import *
from tqdm import tqdm 
import pickle
import os
import torch
import random

folder = "data/nonlinear with current source"
data_num = {"train": 100000, "val": 10000}

for iter in ["train","val"]:
    filepath = folder+"/"+iter
    os.makedirs(filepath, exist_ok=True)
    pbar = tqdm(total=data_num[iter],desc="Generating " + iter + " data")
    while pbar.n < data_num[iter]:
        grid_circuit = GridCircuit(rows=4,cols=4,is_linear=False,has_current_source=True) # makes a random grid circuit each time
        data = grid_circuit.export()
        success, _, record = grid_circuit.solve(convergence_RMS=1e-8,suppress_warning=True)
        if success:
            variance = torch.var(grid_circuit.voltages)
            if variance > 0:
                filename = f"sample_{pbar.n}.pkl"
                with open(os.path.join(filepath,filename), "wb") as f:
                    random_iteration = random.choice(record)
                    output = {"data": data, "answer": grid_circuit.voltages, "RMS_record": random_iteration["RMS"], "x_record": torch.tensor(random_iteration["x"]), "delta_x_record": torch.tensor(random_iteration["delta_x"])}
                    pickle.dump(output, f)
                pbar.update(1)