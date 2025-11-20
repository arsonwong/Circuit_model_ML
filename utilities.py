import yaml

def get_data_generation_settings():
    with open("data_generation_settings.yaml", "r") as f:
        settings = yaml.safe_load(f)
        zeroth_iteration = settings["zeroth_iteration"]
        is_linear = settings["is_linear"]
        has_current_source = settings["has_current_source"]
        acceptable_initial_cond_num = settings["acceptable_initial_cond_num"]
    return zeroth_iteration, is_linear, has_current_source, acceptable_initial_cond_num

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
