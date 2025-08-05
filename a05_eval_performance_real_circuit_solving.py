import torch
from torch.utils.data import DataLoader  # for batching graphs
from tqdm import tqdm  # for progress bar
import pickle
import torch_geometric as pyg
from Circuit_Model_ML.circuit_network_ML import *
from Circuit_Model_ML.circuit_network import *
import os
from a03_training import *
from a02_generate_training_data import get_data_folder

if __name__ == "__main__":
    zeroth_iteration, is_linear, has_current_source, _ = get_data_generation_settings()
    batch_size = 256
    # historical_runs_timestamp_start = "2025-07-30_11-42-23"  # set to None if ignore historical runs
    # historical_runs_timestamp_end = "2025-07-30_13-48-41"  # set to None if ignore historical runs
    # historical_runs_timestamp_start = "2025-07-31_06-42-05"  # set to None if ignore historical runs
    # historical_runs_timestamp_end = "2025-07-31_11-46-27"  # set to None if ignore historical runs

    historical_runs_timestamp_start = "2025-07-30_11-42-23"  # set to None if ignore historical runs
    historical_runs_timestamp_end = "2025-07-31_11-46-27"  # set to None if ignore historical runs
    params = []
    if historical_runs_timestamp_start is not None:
        start_dt = datetime.strptime(historical_runs_timestamp_start, "%Y-%m-%d_%H-%M-%S")
        end_dt = None
        if historical_runs_timestamp_end is not None:
            end_dt = datetime.strptime(historical_runs_timestamp_end, "%Y-%m-%d_%H-%M-%S")

        for folder_name in os.listdir("weights"):
            weight_path = os.path.join("weights", folder_name)
            if not os.path.isdir(weight_path):
                continue
            try:
                folder_dt = datetime.strptime(folder_name, "%Y-%m-%d_%H-%M-%S")
            except ValueError:
                continue  # Skip folders that don't match timestamp format

            if folder_dt >= start_dt and (end_dt is None or folder_dt <= end_dt):
                param_file = os.path.join(weight_path, "parameters.json")
                loss_log_file = os.path.join(weight_path, "loss_log.txt")
                if os.path.isfile(param_file) and os.path.isfile(loss_log_file):
                    with open(param_file, "r", encoding="utf-8") as f:
                        param = json.load(f)
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
                        argmin = np.argmin(val_loss)
                        best_weight_file = os.path.join(weight_path,f"weight_{argmin+1}.pt")
                    param["best_weight_file"] = best_weight_file
                    param["weight_path"] = weight_path
                params.append(param)

    cn = CircuitNetwork().to(device := torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    for param in params:
        folder = get_data_folder(param["is_linear"], param["has_current_source"], param["zeroth_iteration"])
        val_dataset = Dataset(folder,"val2",size=300)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=custom_collate_fn)
        hidden_size = param["hidden_size"]
        n_mp_layers = param["n_mp_layers"]
        model = LearnedSimulator(hidden_size=hidden_size,n_mp_layers=n_mp_layers).to(device := torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        model.load_state_dict(torch.load(param["best_weight_file"]))
        model.eval()
        pbar = tqdm(total=len(val_dataset))
        torch.set_grad_enabled(False)
        solver_num_iterations = torch.zeros(12,dtype=torch.float).to(device)
        solver_num_success = torch.zeros(12,dtype=torch.float).to(device)
        sample_count = 0
        node_count = 0
        for batch in val_loader:
            batch = batch.to(device)

            output, initial_node_error = model(batch)
            node_error = cn(batch,batch.answer)
            answer = batch.answer - batch.x_record # final answer - initial guess
            x_record = batch.x_record
            delta_x_record = batch.delta_x_record
            delta_x_altered_record = batch.delta_x_altered_record

            for j in range(torch.max(batch.sample_indices)):
                find_ = torch.where(batch.sample_indices==j)[0]
                min_index = min(find_)
                max_index = max(find_)
                sample_x_record = batch.x_record[find_,:]
                find2_ = torch.where((batch.edge_index[0,:]>=min_index) & (batch.edge_index[0,:]<=max_index))[0]
                sample_edge_index = batch.edge_index[:,find2_]-min_index
                sample_edge_attr = batch.edge_attr[find2_,:]
                sample_diode_nodes_tensor = batch.diode_nodes_tensor[find2_,:]
                find3_ = torch.where(sample_diode_nodes_tensor[:,0]>=0)[0]
                sample_diode_nodes_tensor[find3_,:] -= min_index
                sample_y = batch.y[find_,:]
                sample_output = output[find_,:]
                sample_answer = answer[find_,:]
                sample_delta_x_record = delta_x_record[find_,:]
                sample_delta_x_altered_record = delta_x_altered_record[find_,:]
                sample_initial_node_error = initial_node_error[find_,:]
                sample_node_error = node_error[find_,:]

                J = torch.autograd.functional.jacobian(lambda x_: cn(pyg.data.Data(x=x_, 
                                                                        edge_index=sample_edge_index, 
                                                                        edge_attr=sample_edge_attr, 
                                                                        diode_nodes_tensor=sample_diode_nodes_tensor,
                                                                        y=sample_y)), sample_x_record).squeeze()
                                
                rows = torch.where(sample_y[:,0]!=1)[0] # not pinned voltage boundary condition
                J_ = J[rows][:, rows]

                sample_output_ = sample_output[rows,:]
                sample_answer_ = sample_answer[rows,:]
                sample_node_error_ = sample_node_error[rows,:]
                sample_delta_x_record_ = sample_delta_x_record[rows,:]
                sample_delta_x_altered_record_ = sample_delta_x_altered_record[rows,:]
                sample_initial_node_error_ = sample_initial_node_error[rows,:]

                RMS = torch.sqrt(torch.sum(sample_node_error_)).item()
                if torch.norm(sample_answer_) > 0 and RMS < 1e-4:
                    if torch.norm(sample_delta_x_record_) ==0 or torch.isinf(sample_delta_x_record_).any():
                        sample_delta_x_record_ = torch.ones_like(sample_output_).to(device)
                    if torch.norm(sample_initial_node_error_) ==0:
                        sample_initial_node_error_ = torch.ones_like(sample_output_).to(device)

                    # naive_search direction = sample_initial_node_error_
                    Y_ = -sample_initial_node_error_.double()
                    # print(Y_)
                    p_ = sample_initial_node_error_.double()
                    alpha = torch.sum((J_ @ p_) * Y_)/torch.sum((J_ @ p_)**2)
                    gradient_guess_delta_x = p_*alpha

                    # CG 1st iteration
                    alpha = (Y_.T @ Y_)/(p_.T @ J_ @ p_)
                    CG_guess_delta_x = p_*alpha

                    # linear search direction best guess
                    p_ = sample_delta_x_record_

                    linear_guess_delta_x = sample_delta_x_record_
                    linear_guess_delta_x_altered = sample_delta_x_altered_record_

                    if param["loss_fn"] > 0:
                        ML_guess_delta_x = sample_output_.double()
                    else:
                        if torch.norm(sample_output_) == 0:
                            sample_output_ = torch.ones_like(sample_output_).to(device)
                        p_ = sample_output_.double()
                        alpha = torch.sum((J_ @ p_) * Y_)/torch.sum((J_ @ p_)**2)
                        ML_guess_delta_x = p_*alpha 

                    data = pyg.data.Data(x=sample_x_record, 
                                        edge_index=sample_edge_index, 
                                        edge_attr=sample_edge_attr, 
                                        diode_nodes_tensor=sample_diode_nodes_tensor,
                                        y=sample_y)
                    
                    guesses = [ML_guess_delta_x, linear_guess_delta_x, linear_guess_delta_x_altered, gradient_guess_delta_x, sample_x_record[rows]]
                    count = 0
                    for method in ["clamp","no_clamp"]:
                        for guess in guesses:
                                data.x[rows] = guess
                                if method=="clamp":
                                    diode_V_pos_delta_limit=0.5
                                    diode_V_hard_limit=0.8
                                else:
                                    diode_V_pos_delta_limit=1000
                                    diode_V_hard_limit=10000
                                                  
                                success, voltages, num_iterations = solve_circuit(data,diode_V_pos_delta_limit=diode_V_pos_delta_limit,
                                                                                diode_V_hard_limit=diode_V_hard_limit,
                                                                                return_num_iterations=True,suppress_warning=True,max_log_diode_I=5)
                                if success:
                                    solver_num_iterations[count] += num_iterations
                                    solver_num_success[count] += 1
                                count += 1

                    if torch.norm(sample_output_) == 0:
                        sample_output_ = torch.ones_like(sample_output_).to(device)

                pbar.update(1)

        avg_solver_num_iterations = solver_num_iterations/solver_num_success
        performance = {
            "avg_solver_num_iterations": avg_solver_num_iterations.tolist(),
            "solver_num_success": solver_num_success.tolist()
            }

        with open(os.path.join(param["weight_path"],"circuit_solving_performance.json"), "w", encoding="utf-8") as f:
            json.dump(performance, f, indent=2)

