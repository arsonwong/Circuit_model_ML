# next ideas:
# distinguish between cold start and non cold start
# starting point isn't zero, but rather the average of all boundary nodes
# detect whether diode has turned on, or has diode
# problem: training set doesn't have overshoots of diodes because that is suppressed
# soln: in rollout we also have the same limits

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader  # for batching graphs
from torch_geometric.data import Batch
from tqdm import tqdm  # for progress bar
import pickle
import torch_geometric as pyg
from Circuit_Model_ML.circuit_network_ML import *
from Circuit_Model_ML.circuit_network import *
import os
import re
from datetime import datetime
import json
from a02_generate_training_data import generate_data

epochs = 100
current_epoch = -1
batch_size = 256
learning_rate = 1e-4

def custom_collate_fn(data_list):
    # Standard PyG batching (handles x, edge_index, edge_attr, y, etc.)
    batch = Batch.from_data_list(data_list)

    # Manually batch diode_nodes_tensor by adding node offset per graph
    diode_nodes = []
    edge_indices = []
    sample_indices = []
    node_offset = 0

    for i, data in enumerate(data_list):
        a = data.diode_nodes_tensor.clone()
        find_ = torch.where(a[:,0]>=0)[0]
        a[find_,:] += node_offset
        diode_nodes.append(a)

        a = data.edge_index.clone()
        a += node_offset
        edge_indices.append(a)

        sample_index = i*torch.ones(data.diode_nodes_tensor.shape[0],1,dtype=torch.long,device=data.diode_nodes_tensor.device)
        sample_indices.append(sample_index)
        node_offset += data.num_nodes

    batch.diode_nodes_tensor = torch.cat(diode_nodes, dim=0)
    batch.edge_index = torch.cat(edge_indices,dim=1)
    batch.sample_indices = torch.cat(sample_indices,dim=0)

    return batch

class Dataset(pyg.data.Dataset):
    def __init__(self, data_path, split, single_example=False):
        super().__init__()
        self.data_path = data_path
        self.split = split
        self.single_example = single_example
        max_size = max([int(m.group(1)) for m in (re.match(r"sample_(\d+)\.pkl", f) for f in os.listdir(os.path.join(self.data_path, self.split))) if m] or [-1])+1
        self.data_set = []
        for i in tqdm(range(max_size), desc=f"Loading {split} dataset"):
            file_path = f"{self.data_path}/{self.split}/sample_{i}.pkl"
            if not os.path.exists(file_path):
                continue
            with open(f"{self.data_path}/{self.split}/sample_{i}.pkl", "rb") as f:
                package = pickle.load(f)
            for instance in package:
                data = instance["data"]
                answer = instance["answer"]
                if "x_record" in instance:
                    data.x_record = instance["x_record"].clone().float()
                    data.delta_x_record = instance["delta_x_record"].clone().float()
                    data.RMS_record = torch.tensor(instance["RMS_record"]).float()
                data.answer = answer.clone().float()
                data.x = data.x_record.clone()
                # data.x = torch.zeros(data.y.shape[0],1,dtype=torch.float) #data.y.clone() #.float()
                data.y = data.y.float()
                data.edge_attr = data.edge_attr.float()
                self.data_set.append(data)

    def len(self):
        return len(self.data_set)
    
    def get(self, idx):
        idx_ = 0
        if self.single_example is False:
            idx_ = idx
        return self.data_set[idx_]

def weight_file_name(path,current_epoch,supervised,single_example):
    return f"{path}/model_epoch_{current_epoch}_supervised={supervised}_single_example={single_example}.pt"

def train(zeroth_iteration = True, is_linear=False, has_current_source=True, hidden_size=128, n_mp_layers=10):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    weight_path = "weights/" + timestamp
    os.makedirs(weight_path, exist_ok=True)
    json_ = {"zeroth_iteration": zeroth_iteration, "is_linear": is_linear, "has_current_source": has_current_source, "hidden_size": hidden_size, "n_mp_layers": n_mp_layers}
    json_path = os.path.join(weight_path, "parameters.json")
    with open(json_path, "w") as f:
        json.dump(json_, f, indent=4)

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

    cn = CircuitNetwork()
    train_dataset = Dataset(folder,"train")
    val_dataset = Dataset(folder,"val")

    # ==== Model, loss, optimizer ====
    model = LearnedSimulator(hidden_size=hidden_size,n_mp_layers=n_mp_layers).to(device := torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params}")
    if current_epoch >= 0:
        model.load_state_dict(torch.load(weight_file_name(weight_path,current_epoch,True,False)))
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    train_dataset.single_example = False
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=custom_collate_fn)

    # ==== Training Loop ====
    val_loss_history = []
    for epoch in range(current_epoch,epochs):
        for train_val in ["train","val"]:
            if train_val=="train":
                model.train()
                progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
            else:
                model.eval()
                progress_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs}")

            torch.set_grad_enabled(True)
            if train_val=="train":
                optimizer.zero_grad()
            else:
                torch.set_grad_enabled(False)
            count = 0
            accum_loss = 0
            total_loss = 0
            sample_count = 0
            for batch in progress_bar:
                batch = batch.to(device)

                output, _ = model(batch)
                answer = batch.answer - batch.x_record # final answer - initial guess

                indices = torch.where(batch.y[:,0]!=1) # not pinned voltage boundary condition
                sample_indices_ = batch.sample_indices[indices]
                output_ = output[indices]
                answer_ = answer[indices]

                for j in range(torch.max(sample_indices_)):
                    find_ = torch.where(sample_indices_==j)
                    sample_output_ = output_[find_]
                    sample_answer_ = answer_[find_]
                    if torch.norm(sample_output_) > 0 and torch.norm(sample_answer_) > 0:
                        answer_direction = sample_answer_ / torch.norm(sample_answer_)
                        output_direction = sample_output_ / torch.norm(sample_output_)
                        accum_loss += torch.sum((answer_direction-output_direction)**2) # deviation from label
                        total_loss += torch.sum((answer_direction-output_direction)**2) 
                        sample_count += 1
                        count += 1

                if train_val=="train":
                    if True: #count==batch_size:
                        accum_loss.backward()
                        optimizer.step()
                        optimizer.zero_grad()
                        count = 0
                        accum_loss = 0

                progress_bar.set_description(f"Epoch {epoch+1}/{epochs}, loss = {total_loss.item()/sample_count:.3e}")

            avg_loss = total_loss.item() / sample_count
            if train_val == "val":
                val_loss_history.append(avg_loss)
            print(f"[Epoch {epoch+1}] Loss: {avg_loss:.4e}")
            with open(os.path.join(weight_path,"loss_log.txt"), "a") as f:
                f.write(f"Epoch {epoch+1} | {train_val} | Loss: {avg_loss}\n")
        torch.save(model.state_dict(), os.path.join(weight_path,f"weight_{epoch+1}.pt"))
        val_loss_history_ = np.array(val_loss_history)
        if val_loss_history_.size > 6:
            part_min_ = np.min(val_loss_history[:-4])
            all_min_ = np.min(val_loss_history)
            if all_min_ > part_min_*0.99:
                break

if __name__ == "__main__":
    for zeroth_iteration in [True, False]:
        if zeroth_iteration:
            generate_data(zeroth_iteration = True)
        for hidden_size in [128,32,64]:
            for n_mp_layers in [10, 5, 20]:
                train(zeroth_iteration=zeroth_iteration, hidden_size=hidden_size, n_mp_layers=n_mp_layers)