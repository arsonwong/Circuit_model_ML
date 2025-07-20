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

# ==== Hyperparameters ====
epochs = 100
current_epoch = -1
batch_size = 256
learning_rate = 1e-6
weight_path = "weights/"

def custom_collate_fn(data_list):
    # Standard PyG batching (handles x, edge_index, edge_attr, y, etc.)
    batch = Batch.from_data_list(data_list)

    # Manually batch diode_nodes_tensor by adding node offset per graph
    diode_nodes = []
    edge_indices = []
    node_offset = 0

    for data in data_list:
        a = data.diode_nodes_tensor.clone()
        find_ = torch.where(a[:,0]>=0)[0]
        a[find_,:] += node_offset
        diode_nodes.append(a)

        a = data.edge_index.clone()
        a += node_offset
        edge_indices.append(a)

        node_offset += data.num_nodes

    batch.diode_nodes_tensor = torch.cat(diode_nodes, dim=0)
    batch.edge_index = torch.cat(edge_indices,dim=1)

    return batch

class Dataset(pyg.data.Dataset):
    def __init__(self, data_path, split, single_example=False):
        super().__init__()
        self.data_path = data_path
        self.split = split
        self.single_example = single_example
        max_size = len([f for f in os.listdir(f"{self.data_path}/{self.split}")])
        self.data_set = []
        for i in tqdm(range(max_size), desc=f"Loading {split} dataset"):
            file_path = f"{self.data_path}/{self.split}/sample_{i}.pkl"
            if not os.path.exists(file_path):
                break
            with open(f"{self.data_path}/{self.split}/sample_{i}.pkl", "rb") as f:
                instance = pickle.load(f)
            data = instance["data"]
            answer = instance["answer"]
            data.answer = answer.clone().float()
            data.x = data.y.clone().float()
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

if __name__ == "__main__":
    cn = CircuitNetwork()
    train_dataset = Dataset("data","train")
    val_dataset = Dataset("data","val")
    for supervised in [True,False]:
        for single_example in [False,True]:
            # ==== Model, loss, optimizer ====
            # model = LearnedSimulator(hidden_size=128,n_mp_layers=10).to(device := torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
            model = LearnedSimulator(hidden_size=32,n_mp_layers=20).to(device := torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
            total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"Total trainable parameters: {total_params}")
            if current_epoch >= 0:
                model.load_state_dict(torch.load(weight_file_name(weight_path,current_epoch,supervised,single_example)))
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            train_dataset.single_example = single_example
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=custom_collate_fn)

            # ==== Training Loop ====
            for epoch in range(current_epoch,epochs):
                for train_val in ["train","val"]:
                    node_count = 0
                    total_loss = np.zeros(4,)
                    if train_val=="train":
                        model.train()
                        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
                    else:
                        model.eval()
                        progress_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs}")

                    torch.set_grad_enabled(True)
                    for batch in progress_bar:
                        batch = batch.to(device)

                        if train_val=="train":
                            optimizer.zero_grad()
                        else:
                            torch.set_grad_enabled(False)

                        output,node_error = model(batch)
                        answer = batch.answer

                        if node_error.dim() == 2 and node_error.shape[1] == 1:
                            node_error = node_error.squeeze(1)
                        # loss1 = torch.sum(node_error**2) # continuity
                        find_ = torch.where(batch.x[:,0]==1)[0] # voltage pinned nodes
                        loss1 = torch.sum(node_error**2) + torch.sum((answer[find_]-output[find_])**2) # continuity + pinned voltage closeness
                        loss2 = torch.sum((answer-output)**2) # deviation from label

                        if train_val=="train":
                            if supervised:
                                loss2.backward()
                            else:
                                loss1.backward()
                            optimizer.step()

                        data = pyg.data.Data(
                            x=batch.answer,  
                            edge_index=batch.edge_index,  
                            edge_attr=batch.edge_attr,
                            diode_nodes_tensor=batch.diode_nodes_tensor, 
                            y=batch.y
                        )
                        node_error = cn.forward(data)
                        print(batch.answer)
                        print(output)
                        print(node_error)
                        assert(1==0)

                        if node_error.dim() == 2 and node_error.shape[1] == 1:
                            node_error = node_error.squeeze(1)
                        loss3 = torch.sum(node_error**2) # continuity from the answer

                        data = pyg.data.Data(
                            x=output,  
                            edge_index=batch.edge_index,  
                            edge_attr=batch.edge_attr, 
                            y=batch.y,
                            diode_nodes_tensor=batch.diode_nodes_tensor
                        )
                        node_error = cn.forward(data)
                        if node_error.dim() == 2 and node_error.shape[1] == 1:
                            node_error = node_error.squeeze(1)
                        loss4 = torch.sum(node_error**2) # continuity from this output, should be identical to loss1
                        node_count += node_error.numel()

                        total_loss[0] += loss1.item()
                        total_loss[1] += loss2.item()
                        total_loss[2] += loss3.item()
                        total_loss[3] += loss4.item()
                        progress_bar.set_description(f"Epoch {epoch+1}/{epochs}, loss = {loss2.item()/node_error.numel():.3e}")

                    avg_loss = total_loss / node_count
                    print(f"[Epoch {epoch+1}] Loss: {avg_loss[1]:.4e}")
                    with open(f"loss_log_supervised={supervised}_single_example={single_example}.txt", "a") as f:
                        f.write(f"Epoch {epoch+1} | {train_val} | Losses: {avg_loss.tolist()}\n")
                os.makedirs(weight_path, exist_ok=True)
                torch.save(model.state_dict(), weight_file_name(weight_path,epoch+1,supervised,single_example))