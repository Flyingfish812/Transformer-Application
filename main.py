from presets import *
from my_utils import *
import sys
import time as t
import torch
import torch.nn as nn
import torch.optim as optim

def main(config_path):
    print('----- Training Session -----')
    # Read config
    print('Reading Config ... ', end='')
    config = read_config(config_path)
    print('Complete')
    print(f'Model Name: {config["model"]["name"]}')
    print(f'Expected sessions: {25*config["training"]["batch_size"]}')

    # Load data
    print('Loading Data ... ', end='')
    lat, lon, sst_all, time = load_data(config['data']['file_path'])
    print('Complete')

    # Build model
    print('Building Model ... ', end='')
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
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    print('Complete')

    # Training and testing logic
    data_loader = data_geneator(lat, lon, time, sst_all, 
                                batch_size = config['training']['batch_size'],
                                device = device,
                                start_point = config['training']['start_point'],
                                sensor_num = config['training']['sensor_num'],
                                sensor_seed = config['training']['sensor_seed'],
                                sigma = config['training']['sigma'],
                                inputType = config['training']['input_type'])
    start_time = t.time()
    train(model, data_loader, optimizer, criterion, method='SWA', swa_start=10)
    end_time = t.time()
    exec_time = end_time - start_time
    print(f'Training Complete in {exec_time}')

    # Save model and results
    print('Saving Model ... ', end='')
    torch.save(model.state_dict(), config['output']['model_save_path'])
    print('Complete')


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python main.py config.yaml")
        sys.exit(1)
    config_path = sys.argv[1]  # Get the configuration file path from the command line
    main(config_path)