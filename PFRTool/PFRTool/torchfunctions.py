import torch
import torch.nn as nn
import numpy as np
from .inputs import *
from .builder import *

'''
A function to pad a 2D array to make it square.
Attention, this function is used only for numpy arrays, so it should be used before turing it to a torch tensor.
'''
def make_square(array):
    m, n = array.shape  # Get the dimensions of the array
    
    if m == n:
        return array  # Return the array as is if it is already square
    
    # Determine padding amounts
    if m < n:
        # More columns than rows, pad vertically
        padding = (n - m) // 2
        extra = (n - m) % 2
        pad_width = ((padding + extra, padding), (0, 0))
    else:
        # More rows than columns, pad horizontally
        padding = (m - n) // 2
        extra = (m - n) % 2
        pad_width = ((0, 0), (padding + extra, padding))
    
    # Pad with zeros and return
    return np.pad(array, pad_width, mode='constant')

'''
What should be included in the infolist to build the setup list:
    - model_config
    - (optional) method
    - (optional) input_type
    - (optional) criterion
    - (optional) optimizer
    - (optional) learning_rate
    - (optional) device
    - (optional) train_epoch
    - (optional) step_size
The output setup list will include these parameters:
    - model
    - method
    - input_type
    - criterion
    - optimizer
    - scheduler
    - device
    - train_epoch
    - step_size
'''
def setup_info(infolist):
    setup = {}
    if 'model_config' in infolist:
        model = build_model(infolist['model_config'])
        if(infolist['model_config']['load_weights']):
            model.load_state_dict(torch.load(infolist['model_save_path']))
        model.to(infolist['device'])
        setup['model'] = model
    else:
        raise ValueError("'model_config' is missing")
    if 'method' in infolist:
        setup['method'] = infolist['method']
    else:
        setup['method'] = 'base'
    if 'input_type' in infolist:
        setup['input_type'] = infolist['input_type']
    else:
        setup['input_type'] = 'VIT'
    if 'criterion' in infolist:
        setup['criterion'] = infolist['criterion']
    else:
        setup['criterion'] = nn.MSELoss()
    if 'optimizer' in infolist and 'scheduler' in infolist:
        setup['optimizer'] = infolist['optimizer']
        setup['scheduler'] = infolist['scheduler']
    else:
        optimizer, scheduler = build_optimizer(model, initial_lr=infolist['learning_rate'])
        setup['optimizer'] = optimizer
        setup['scheduler'] = scheduler
    if 'device' in infolist:
        setup['device'] = infolist['device']
    else:
        setup['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    if 'train_epoch' in infolist:
        setup['train_epoch'] = infolist['train_epoch']
    else:
        setup['train_epoch'] = 3
    if 'step_size' in infolist:
        setup['step_size'] = infolist['step_size']
    else:
        setup['step_size'] = 16
    return setup

def to_input(source, setup):
    if(setup['input_type'] == "VIT"):
        source = make_square(source)
    tensor = torch.from_numpy(source.copy())    # To a tensor
    tensor = tensor.to(torch.float32)
    tensor = tensor.to(setup['device'])    # To specific device
    return tensor

# If you want to get one sample to visualize
def get_one_sample(STUitem: STU, setup):
    source = to_input(STUitem.input, setup)
    source_map = to_input(STUitem.infolist['map_location'], setup)
    target = torch.from_numpy(STUitem.output.copy()).to(setup['device']).unsqueeze(0)
    combined_source = torch.stack([source.unsqueeze(0), source_map.unsqueeze(0), source.unsqueeze(0) * source_map.unsqueeze(0)], dim=1)
    return target, combined_source

# Function for cross validation, use swap to ensure that each 10% data can be the validation set.
def n_swap(arr, n):
    num_elements = len(arr)
    swap_size = int(num_elements * n * 0.1)
    last_part = arr[-swap_size:]
    first_part = arr[:-swap_size]
    return np.concatenate([last_part, first_part])

'''
Dataset generate function
What should be included in the setup:
    - input_type
    - device
What should be included in the infolist:
    - total_fignum
    - batch_size
    - sensor_num
    - sensor_seed
    - (optional) only_test
    - (optional) cross_validation_num
'''
def data_generator(input_set, output_set, setup, infolist):
    # Basic settings
    only_test = infolist['only_test'] if 'only_test' in infolist else False
    cross_validation_num = infolist['cross_validation_num'] if 'cross_validation_num' in infolist else 0
    total_samples = infolist['total_fignum'] // infolist['batch_size']
    train_cutoff = int(0.9 * total_samples)
    verify_cutoff = train_cutoff + int(0.1 * total_samples)

    # Initialization
    current_sample = 0
    targets, sources = [], []
    fig_num_list = list(range(infolist['total_fignum']))
    np.random.seed(0)
    np.random.shuffle(fig_num_list)
    fig_num_list = n_swap(fig_num_list, cross_validation_num)

    for i in fig_num_list:
        i = int(i)
        STUitem = STU(input=input_set[i], output=output_set[i], infolist=infolist)
        STUitem.set_trainer()
        target, source = get_one_sample(STUitem, setup)
        if only_test:
            category = 'test'
        elif current_sample < train_cutoff:
            category = 'train'
        elif current_sample < verify_cutoff:
            category = 'verify'
        else:
            category = 'test'

        targets.append(target)
        sources.append(source)

        # Check if the batch has reached the specified batch size
        if len(targets) == infolist['batch_size']:
            yield (torch.cat(targets, dim=0), torch.cat(sources, dim=0), category)
            current_sample += 1
            targets, sources = [], []

    # After the loop, yield any remaining samples in the batch
    if targets:
        yield (torch.cat(targets, dim=0), torch.cat(sources, dim=0), category)
        current_sample += 1
        targets, sources = [], []

'''
Factory function is used to generate the data generator. It takes 3 parameters:
    - variables: it contains the input_set and the output_set, read from external files
    - setup: it contains the basic settings
    - infolist: it contains the configurations
The requirements of the setup and the infolist are listed in the above function.
'''
def factory(variables, setup, infolist):
    input_set = variables['input_set']
    output_set = variables['output_set']
    data_gen = data_generator(input_set, output_set, setup, infolist)
    return data_gen

'''
Norm function
Attention, this function is only available for torch tensor objects.
'''
def calculate_norm(output, target, type="L2"):
    # L2 relative loss
    if type == "L2":
        loss = torch.norm(output - target, p=2)
        norm_val = torch.norm(target, p=2)
        error_val = loss / norm_val

    # L infinity relative loss
    elif type == "Linf":
        loss = torch.max(torch.abs(output - target))
        norm_val = torch.max(target)
        error_val = loss / norm_val
    
    return error_val.item()