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
from a03_training import *
import re

if __name__ == "__main__":
    cn = CircuitNetwork()
    # ==== Model, loss, optimizer ====
    model = LearnedSimulator(hidden_size=hidden_size,n_mp_layers=n_mp_layers).to(device := torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    pattern = r"model_epoch_(\d+)_supervised=True_single_example=False\.pt"
    max_epoch = -1
    for fname in os.listdir(weight_path):
        match = re.match(pattern, fname)
        if match:
            epoch = int(match.group(1))
            if epoch > max_epoch:
                max_epoch = epoch
    model.load_state_dict(torch.load(weight_file_name(weight_path,max_epoch,True,False)))

    val_dataset = Dataset("data/nonlinear with current source","val")
    val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=custom_collate_fn)

    model.eval()
    progress_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs}")
    torch.set_grad_enabled(False)
    total_loss = torch.tensor([0,0,0],dtype=torch.float).to(device)
    sample_count = 0
    for batch in progress_bar:
        batch = batch.to(device)

        output, initial_node_error = model(batch)
        answer = batch.answer
        x_init_record = batch.x_init_record
        delta_x_init_record = batch.delta_x_init_record

        indices = torch.where(batch.y[:,0]!=1) # not pinned voltage boundary condition
        sample_indices_ = batch.sample_indices[indices]
        output_ = output[indices]
        answer_ = answer[indices]
        delta_x_init_record_ = delta_x_init_record[indices]
        initial_node_error_ = initial_node_error[indices]

        for j in range(torch.max(sample_indices_)):
            find_ = torch.where(sample_indices_==j)
            sample_output_ = output_[find_]
            sample_answer_ = answer_[find_]
            sample_delta_x_init_record_ = delta_x_init_record_[find_]
            sample_initial_node_error_ = initial_node_error_[find_]
            if torch.norm(sample_answer_) > 0:
                if torch.norm(sample_output_) == 0:
                    sample_output_ = torch.ones_like(sample_output_).to(device)
                if torch.norm(sample_delta_x_init_record_) ==0:
                    sample_delta_x_init_record_ = torch.ones_like(sample_output_).to(device)
                if torch.norm(sample_initial_node_error_) ==0:
                    sample_initial_node_error_ = torch.ones_like(sample_output_).to(device)
                answer_direction = sample_answer_ / torch.norm(sample_answer_)
                output_direction = sample_output_ / torch.norm(sample_output_)
                linear_direction = sample_delta_x_init_record_ / torch.norm(sample_delta_x_init_record_)
                gradient_direction = sample_initial_node_error_ / torch.norm(sample_initial_node_error_)
                total_loss[0] += torch.sum((answer_direction-output_direction)**2) 
                total_loss[1] += torch.sum((answer_direction-gradient_direction)**2) 
                total_loss[2] += torch.sum((answer_direction-linear_direction)**2) 
                sample_count += 1

        progress_bar.set_description(f"Epoch {epoch+1}/{epochs}, losses = {total_loss[0].item()/sample_count:.3e}, {total_loss[1].item()/sample_count:.3e}, {total_loss[2].item()/sample_count:.3e}")