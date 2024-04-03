from presets import *
from my_utils import *
import sys
import time as t
import torch
import torch.nn as nn
import torch.optim as optim

def main(config_path):
    print('----- Testing Session -----')
    # Read config
    print('Reading Config ... ', end='')
    config = read_config(config_path)
    session_num = round(config["testing"]["fig_num"]*len(config["training"]["sensor_num"]+config["testing"]["sensor_num"])*
                   (len(config["testing"]["sensor_seed"]+config["training"]["sensor_seed"]))/
                   config["testing"]["batch_size"])
    print('Complete')
    print(f'Model Name: {config["model"]["name"]}')
    print(f'Expected sessions: {session_num}')

    # Load data
    print('Loading Data ... ', end='')
    lat, lon, sst_all, time = load_data(config['data']['file_path'])
    print('Complete')

    # Read model
    print('Reading Model ... ', end='')
    model = build_model(config['model'])
    model.load_state_dict(torch.load(config['output']['model_save_path']))
    print('Complete')

    # Additional setup: optimizer, criterion, device
    print('Additional Setup ... ', end='')
    if(config['device']['prefer_gpu']):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    model.to(device)

    batch_size = config['testing']['batch_size']
    norm = config['testing']['norm']
    print('Complete')

    # Testing
    data_config = {"lat": lat, "lon": lon, "time": time, "sst_all": sst_all,
                   "fig_num": config['testing']['fig_num'],
                   "batch_size": batch_size, "device": device,
                   "start_point": config['testing']['start_point'],
                   "sensor_num": config['testing']['sensor_num'],
                   "sensor_seed": config['testing']['sensor_seed'],
                   "sigma": config['testing']['sigma'],
                   "inputType": config['testing']['input_type']
                   }
    start_time = t.time()
    testing_data = test(model, data_config, device, norm=norm)
    end_time = t.time()
    exec_time = end_time - start_time
    print(f'Testing Complete in {exec_time:.2f} seconds')

    # Save the results
    print('Saving Result ... ', end='')
    dump_result(config, testing_data, exec_time, mode="test")
    print('Complete')


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python test.py config.yaml")
        sys.exit(1)
    config_path = sys.argv[1]  # Get the configuration file path from the command line
    main(config_path)
