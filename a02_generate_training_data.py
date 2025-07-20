from Circuit_Model_ML.circuit_network import *
from tqdm import tqdm 
import pickle
import os

folder = "data"
data_num = {"train": 100000, "val": 10000}

for iter in ["train","val"]:
    filepath = folder+"/"+iter
    os.makedirs(filepath, exist_ok=True)
    pbar = tqdm(total=data_num[iter],desc="Generating " + iter + " data")
    while pbar.n < data_num[iter]:
        grid_circuit = GridCircuit(rows=4,cols=4,is_linear=True,has_current_source=False) # makes a random grid circuit each time
        data = grid_circuit.export()
        success, _ = grid_circuit.solve(convergence_RMS=1e-8,suppress_warning=True)
        if success:
            filename = f"sample_{pbar.n}.pkl"
            with open(os.path.join(filepath,filename), "wb") as f:
                output = {"data": data, "answer": grid_circuit.voltages}
                pickle.dump(output, f)
            pbar.update(1)