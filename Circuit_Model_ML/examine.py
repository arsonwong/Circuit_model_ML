import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import DataLoader  # for batching graphs
from tqdm import tqdm  # for progress bar
from circuit_network import *
from training import *

if __name__ == "__main__":
    cn = CircuitNetwork()

    model = LearnedSimulator(hidden_size=32,n_mp_layers=4).to(device := torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    model.load_state_dict(torch.load(r"../weights/model_epoch_2_supervised=False_single_example=False.pt"))
    train_dataset = Dataset("../data","train",single_example=True)
    val_dataset = Dataset("../data","val")
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1)
    model.eval()
    torch.set_grad_enabled(False)
    progress_bar = tqdm(train_loader)
    for batch in progress_bar:
        output,node_error = model(batch)
        answer = batch.answer

        if node_error.dim() == 2 and node_error.shape[1] == 1:
            node_error = node_error.squeeze(1)
        loss1 = torch.sum(node_error**2) # continuity
        loss2 = torch.sum((answer-output)**2) # deviation from label
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

        # output[1:,0] += 2
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
        node_count = node_error.numel()

        total_loss = np.array([loss1.item(),loss2.item(),loss3.item(),loss4.item()])
        avg_loss = total_loss / node_count

        print(node_error)
        print(avg_loss)
        print(answer)
        print(output)
        assert(1==0)
    