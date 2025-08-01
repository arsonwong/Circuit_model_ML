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
from a02_generate_training_data import get_data_folder
from a01_example_circuit_network import get_data_generation_settings

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

        sample_index = i*torch.ones(data.num_nodes,1,dtype=torch.long,device=data.diode_nodes_tensor.device)
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
        self.aux_set = []
        for i in tqdm(range(max_size), desc=f"Loading {split} dataset"):
            file_path = f"{self.data_path}/{self.split}/sample_{i}.pkl"
            if not os.path.exists(file_path):
                continue
            with open(f"{self.data_path}/{self.split}/sample_{i}.pkl", "rb") as f:
                package = pickle.load(f)
            for instance in package:
                data = instance["data"]
                answer = instance["answer"]
                for aux_info in ["x_record", "delta_x_record", "delta_x_altered_record"]:
                    if aux_info in instance:
                        setattr(data, aux_info, instance[aux_info].clone())
                data.x = data.x_record.clone()
                data.answer = answer.clone()
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

def prep_train_or_ver(zeroth_iteration = True, is_linear=False, has_current_source=True, 
                 hidden_size=128, n_mp_layers=10, layernorm=True, loss_fn=0, starting_point=None, learning_rate = 1e-4):
    if starting_point is None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        weight_path = "weights/" + timestamp
        os.makedirs(weight_path, exist_ok=True)
        json_ = {
        "zeroth_iteration": zeroth_iteration, 
        "is_linear": is_linear, 
        "has_current_source": has_current_source, 
        "hidden_size": hidden_size, 
        "n_mp_layers": n_mp_layers,
        "layernorm": layernorm, 
        "loss_fn": loss_fn
        }
        json_path = os.path.join(weight_path, "parameters.json")
        with open(json_path, "w") as f:
            json.dump(json_, f, indent=4)
    else:
        weight_path = starting_point["start_weight_path"]

    folder = get_data_folder(is_linear, has_current_source, zeroth_iteration)

    # ==== Model, loss, optimizer ====
    model = LearnedSimulator(hidden_size=hidden_size,n_mp_layers=n_mp_layers).to(device := torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    if starting_point is not None:
        model.load_state_dict(torch.load(starting_point["start_weight_filename"]))
        optimizer.load_state_dict(torch.load(os.path.join(weight_path, "optimizer.pth")))

    return model, optimizer, folder, weight_path, loss_fn

def train(model, optimizer, folder, weight_path, loss_fn, epochs = 100, batch_size = 256):
    train_dataset = Dataset(folder,"train")
    val_dataset = Dataset(folder,"val")

    train_dataset.single_example = False
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=custom_collate_fn)

    device = next(model.parameters()).device
    cn = CircuitNetwork(max_log_diode_I = 5).to(device)

    # ==== Training Loop ====
    val_loss_history = []
    train_loss_history = []
    start_epoch = 0
    if starting_point is not None:
        start_epoch = starting_point["start_epoch"]
        loss_log_file = os.path.join(weight_path, "loss_log.txt")
        if os.path.isfile(loss_log_file):
            with open(loss_log_file, "r") as f:
                val_loss = []
                for line in f:
                    line = line.strip()
                    if "Val Loss:" in line:
                        # Split and get the last value
                        val_str = line.split("Val Loss:")[1].strip()
                        val_loss.append(float(val_str))
                    if "val | Loss:" in line:
                        val_str = line.split("val | Loss:")[1].strip()
                        val_loss.append(float(val_str))
                val_loss_history = val_loss
    for epoch in range(start_epoch,epochs):
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
                node_error = cn(batch,x=output)
                answer = batch.answer - batch.x_record # final answer - initial guess

                indices = torch.where(batch.y[:,0]!=1) # not pinned voltage boundary condition
                sample_indices_ = batch.sample_indices[indices]
                output_ = output[indices]
                answer_ = answer[indices]
                node_error_ = node_error[indices]

                for j in range(torch.max(sample_indices_)):
                    find_ = torch.where(sample_indices_==j)
                    sample_output_ = output_[find_]
                    sample_answer_ = answer_[find_]
                    sample_node_error_ = node_error_[find_]
                    if torch.norm(sample_output_) > 0 and torch.norm(sample_answer_) > 0:
                        answer_mag = torch.norm(sample_answer_)
                        output_mag = torch.norm(sample_output_)
                        answer_direction = sample_answer_ / answer_mag
                        output_direction = sample_output_ / output_mag
                        if loss_fn == 0:
                            this_loss = torch.sum((answer_direction-output_direction)**2) # deviation from label
                        elif loss_fn == 1:
                            this_loss = torch.sum((answer_direction-output_direction)**2) + (torch.log(answer_mag/output_mag))**2 # deviation from label
                        elif loss_fn == 2:
                            this_loss = torch.sum((sample_answer_-sample_output_)**2)
                        else:
                            this_loss = torch.sum(sample_node_error_**2)  
                        accum_loss += this_loss
                        total_loss += this_loss
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
            else:
                train_loss_history.append(avg_loss)
        print(f"[Epoch {epoch+1}] Train Loss: {train_loss_history[-1]:.4e}, Val Loss: {val_loss_history[-1]:.4e}")
        with open(os.path.join(weight_path,"loss_log.txt"), "a") as f:
            f.write(f"[Epoch {epoch+1}] Train Loss: {train_loss_history[-1]:.4e}, Val Loss: {val_loss_history[-1]:.4e}\n")
        torch.save(model.state_dict(), os.path.join(weight_path,f"weight_{epoch+1}.pt"))
        torch.save(optimizer.state_dict(),  os.path.join(weight_path,f"optimizer.pth"))
        
        val_loss_history_ = np.array(val_loss_history)
        if val_loss_history_.size > 12:
            part_min_ = np.min(val_loss_history[:-10])
            all_min_ = np.min(val_loss_history)
            if all_min_ > part_min_*0.99:
                break

    weight_files = [f for f in os.listdir(weight_path) if f.startswith("weight_") and f.endswith(".pt")]
    # Extract the number from weight_{number}.pt and sort numerically
    weight_files = sorted(weight_files, key=lambda x: int(x.split("_")[1].split(".")[0]))
    if len(weight_files) > 12:
        for old_file in weight_files[:-11]:
            os.remove(os.path.join(weight_path, old_file))

if __name__ == "__main__":
    zeroth_iteration, is_linear, has_current_source, _ = get_data_generation_settings()
    historical_runs_timestamp_start = None
    params = []
    val_losses = []
    weight_paths = []
    if historical_runs_timestamp_start is not None:
        ref_dt = datetime.strptime(historical_runs_timestamp_start, "%Y-%m-%d_%H-%M-%S")
        for folder_name in os.listdir("weights"):
            weight_path = os.path.join("weights", folder_name)

            if not os.path.isdir(weight_path):
                continue
            try:
                folder_dt = datetime.strptime(folder_name, "%Y-%m-%d_%H-%M-%S")
            except ValueError:
                continue  
            if folder_dt >= ref_dt:
                param_file = os.path.join(weight_path, "parameters.json")
                loss_log_file = os.path.join(weight_path, "loss_log.txt")
                if os.path.isfile(param_file) and os.path.isfile(loss_log_file):
                    with open(param_file, "r", encoding="utf-8") as f:
                        params.append(json.load(f))
                    with open(loss_log_file, "r") as f:
                        val_loss = []
                        for line in f:
                            line = line.strip()
                            if "Val Loss:" in line:
                                # Split and get the last value
                                val_str = line.split("Val Loss:")[1].strip()
                                val_loss.append(float(val_str))
                            if "val | Loss:" in line:
                                val_str = line.split("val | Loss:")[1].strip()
                                val_loss.append(float(val_str))
                        val_loss = np.array(val_loss)
                        val_losses.append(val_loss.copy())
                    weight_paths.append(weight_path)
                    print(f"--- {folder_name} ---")
                    # print(json.dumps(params, indent=4))

    for hidden_size in [64,128,256]:
        for n_mp_layers in [5,10,20]:
            for layernorm in [True, False]: 
                for loss_fn in [0,1,2,3]:
                    json_ = {
                    "zeroth_iteration": zeroth_iteration, 
                    "is_linear": is_linear, 
                    "has_current_source": has_current_source, 
                    "hidden_size": hidden_size, 
                    "n_mp_layers": n_mp_layers,
                    "layernorm": layernorm, 
                    "loss_fn": loss_fn
                    }
                    started = False
                    completed = False
                    start_weight_path = []
                    start_weight_filename = []
                    start_epoch = -1
                    for j, param in enumerate(params):
                        if param == json_:
                            started = True
                            val_loss_history = val_losses[j]
                            start_weight_path = weight_paths[j]
                            start_epoch = len(val_loss_history)
                            start_weight_filename = os.path.join(start_weight_path,f"weight_{start_epoch}.pt")
                            if val_loss_history.size > 12:
                                part_min_ = np.min(val_loss_history[:-10])
                                all_min_ = np.min(val_loss_history)
                                if all_min_ > part_min_*0.99:
                                    completed = True
                                    break
                    if completed:
                        continue
                    starting_point = None
                    if started:
                        starting_point = {"start_weight_path": start_weight_path, "start_weight_filename": start_weight_filename, "start_epoch": start_epoch}
                    train(*prep_train_or_ver(zeroth_iteration=zeroth_iteration, hidden_size=hidden_size, n_mp_layers=n_mp_layers, 
                                            layernorm=layernorm, loss_fn=loss_fn, starting_point=starting_point,
                                            is_linear=is_linear,has_current_source=has_current_source))
