import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import DataLoader  # for batching graphs
from tqdm import tqdm  # for progress bar
from circuit_network import *
import os

# ==== Hyperparameters ====
epochs = 28
current_epoch = -1
batch_size = 256
learning_rate = 1e-6
weight_path = "../weights/"

def generate_data(tandem_model, data_path, split, num=10000):
    tandem_model_clone = circuit_deepcopy(tandem_model)
    data_ = []
    for i in tqdm(range(num)):
        scales = np.random.lognormal(0.0, 0.2, size=7)
        tandem_model.cells[0].set_I01(tandem_model_clone.cells[0].I01()*scales[0])
        tandem_model.cells[0].set_I02(tandem_model_clone.cells[0].I02()*scales[1])
        tandem_model.cells[0].set_shunt_cond(tandem_model_clone.cells[0].shunt_cond()*scales[2])
        tandem_model.cells[1].set_I01(tandem_model_clone.cells[1].I01()*scales[3])
        tandem_model.cells[1].set_I02(tandem_model_clone.cells[1].I02()*scales[4])
        tandem_model.cells[1].set_shunt_cond(tandem_model_clone.cells[1].shunt_cond()*scales[5])
        tandem_model.set_Rs(tandem_model_clone.Rs()*scales[6])

        target_I = 4
        tandem_model.set_operating_point(I=target_I)
        V1 = tandem_model.operating_point[0]
        V3 = V1-tandem_model.subgroups[2].operating_point[0]
        V2 = V3-tandem_model.subgroups[1].operating_point[0]
        x = torch.tensor([0.0,V1,V2,V3])[:,None]

        coo = translate_to_COO(tandem_model)
        edge_index = coo[:,:2].long().t()
        edge_feature = coo[:,2:]

        data = pyg.data.Data(
            x=x,  
            edge_index=edge_index,  
            edge_attr=edge_feature, 
            y=node_boundary_conditions
        )
        node_error = cn.forward(data)
        rows = torch.where(node_boundary_conditions[:,0] != 1)[0]
        for j in range(7):
            def f(x_):
                data = pyg.data.Data(x=x_, edge_index=edge_index, edge_attr=edge_feature)
                return cn(data)

            J = torch.autograd.functional.jacobian(f, x).squeeze()
            J = J[rows][:, rows]

            node_error = node_error[rows]
            Y = -node_error

            X = torch.linalg.solve(J, Y)

            x[rows] = x[rows] + X

            data = pyg.data.Data(
                x=x, 
                edge_index=edge_index,  
                edge_attr=edge_feature,
                y=node_boundary_conditions
            )

            node_error = cn.forward(data)
        assert(torch.sum(node_error**2).item() < 1e-12*4)
        data_.append((node_boundary_conditions,edge_index,edge_feature,x))
        if i > 0 and i % 10000 == 0:
            # pickle dump the data
            out_path = os.path.join(data_path, f"{split}.pkl")
            with open(out_path, "wb") as f:
                pickle.dump(data_, f)
    # pickle dump the data
    out_path = os.path.join(data_path, f"{split}.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(data_, f)

class Dataset(pyg.data.Dataset):
    def __init__(self, data_path, split, single_example=False):
        super().__init__()
        self.data_path = data_path
        self.split = split
        self.single_example = single_example
        with open(f"{data_path}/{split}.pkl", "rb") as f:
            self.data = pickle.load(f)
    
    def len(self):
        return len(self.data)
    
    def get(self, idx):
        idx_ = 0
        if self.single_example is False:
            idx_ = idx
        node_boundary_conditions,edge_index,edge_feature,x = self.data[idx_]
        graph = pyg.data.Data(
            x=node_boundary_conditions,  
            edge_index=edge_index,  
            edge_attr=edge_feature,
            y=node_boundary_conditions,
            answer = x  
        )
        return graph

def weight_file_name(path,current_epoch,supervised,single_example):
    return f"{path}/model_epoch_{current_epoch}_supervised={supervised}_single_example={single_example}.pt"



if __name__ == "__main__":
    # tandem_model = pickle.load(open(r"C:\Users\arson\Documents\Tandem_Cell_Fit_Tools\best_fit_tandem_model.pkl", 'rb'))
    # generate_data(tandem_model, "../data","train",100000)
    # generate_data(tandem_model, "../data","val",10000)
    # assert(1==0)

    cn = CircuitNetwork()

    for supervised in [False,True]:
        for single_example in [False,True]:
            # ==== Model, loss, optimizer ====
            # model = LearnedSimulator(hidden_size=128,n_mp_layers=10).to(device := torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
            model = LearnedSimulator(hidden_size=32,n_mp_layers=4).to(device := torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
            total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"Total trainable parameters: {total_params}")
            if current_epoch >= 0:
                model.load_state_dict(torch.load(weight_file_name(weight_path,current_epoch,supervised,single_example)))
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)

            train_dataset = Dataset("../data","train",single_example=single_example)
            val_dataset = Dataset("../data","val")
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size)

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
                            y=batch.y
                        )
                        node_error = cn.forward(data)
                        if node_error.dim() == 2 and node_error.shape[1] == 1:
                            node_error = node_error.squeeze(1)
                        loss3 = torch.sum(node_error**2) # continuity from the answer

                        data = pyg.data.Data(
                            x=output,  
                            edge_index=batch.edge_index,  
                            edge_attr=batch.edge_attr, 
                            y=batch.y
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
                torch.save(model.state_dict(), weight_file_name(weight_path,epoch+1,supervised,single_example))