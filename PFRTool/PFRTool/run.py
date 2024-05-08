from .iofunctions import *
from .builder import *
from .torchfunctions import *
from .traintest import *
import time as t

'''
Infolist builder function
    - model_list: used to build the model
        - name
        - model_save_path
    - data_list: used to generate inputs
        - input_source
        - output_source
        - input_name
        - output_name
        - need_sparse
        - *map
        - *sigma
        - lat
        - lon
        - nan
        - sensor_seed
        - sensor_num
        - total_fignum
        - batch_size
        - (optional) only_test
        - (optional) cross_validation_num
    - train_list
        - method
        - *input_type
        - *criterion
        - *optimizer
        - *learning_rate
        - *device
        - *train_epoch
        - *step_size
'''
def build_infolist(config):
    model_list = config['model']
    data_list = config['data']
    train_list = config['train']
    test_list = config['test']
    return model_list, data_list, train_list, test_list

def run(config_path, mode = 'train'):
    global variables

    # 1. Display the session type
    if mode == 'train':
        print('----- Training Session -----')
    elif mode == 'test':
        print('----- Testing Session -----')
    
    # 2. Read the config
    print('Reading Config ... ', end='')
    config = read_config(config_path)
    model_list, data_list, train_list, test_list = build_infolist(config)
    print('Complete')

    session_num = data_list['total_fignum'] // data_list['batch_size'] + 1
    epoch_num = train_list['train_epoch'] if 'train_epoch' in train_list else 3
    print(f'Model Name: {model_list["name"]}')
    print(f'Expected sessions: {session_num}, with {epoch_num} epochs')

    # 3. Read the data
    print('Loading Data ... ')
    length_out = 0
    if data_list['output_source'] != '':
        read_mat(data_list['output_source'])
        variables['output_set'] = variables[data_list['output_name']]
        length_out = len(variables['output_set'])
    else:
        print('No output data')
    if data_list['input_source'] != '':
        read_mat(data_list['input_source'])
        variables['input_set'] = variables[data_list['input_name']]
        length_in = len(variables['input_set'])
        if length_out != length_in:
            raise ValueError('Input and Output data length mismatch')
    else:
        print('No input data, use None to fill')
        variables['input_set'] = [None for _ in range(length_out)]
    print('Complete')

    # 4. Training setup
    print('Model and training Setup ... ', end='')
    temp_list = train_list.copy()
    temp_list['model_config'] = model_list
    setup = setup_info(temp_list)
    model = setup['model']
    print('Complete')

    # 5. Training or testing
    start_time = t.time()
    all_list = {**data_list, **train_list}
    training_data = train(model, setup, all_list)
    end_time = t.time()
    exec_time = end_time - start_time
    print(f'Training Complete in {exec_time:.2f} seconds')

    # 6. Dump the results
    print('Saving Model and getting training patterns... ')
    torch.save(model.state_dict(), model_list['model_save_path'])
    
    dump_result(config, training_data, exec_time, mode="train")
    print('Complete')