from presets import *
from my_utils import *
from ensemble import *
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
    session_num = round(config["training"]["fig_num"]*len(config["training"]["sensor_num"])*len(config["training"]["sensor_seed"])/config["training"]["batch_size"])
    print('Complete')
    print(f'Model Name: {config["model"]["name"]}')
    print(f'Method: {config["model"]["method"]}')
    print(f'Expected sessions: {session_num}, with {config["training"]["num_epochs"]} epochs')

    # Load data
    print('Loading Data ... ', end='')
    lat, lon, sst_all, time = load_data(config['data']['file_path'])
    # models = []
    # for i in range(config['model']['num_models']):
    #     model = build_model(config[f'model_{i+1}'])
    #     model.load_state_dict(torch.load(config[f'model_{i+1}']['path']))
    #     models.append(model)
    print('Complete')

    # Build model
    print('Building Model ... ', end='')
    model = build_ensemble_model(config)
    if(config['model']['load_weights']):
        model.load_state_dict(torch.load(config['output']['model_save_path']))
    print('Complete')

    # Additional setup: optimizer, criterion, device
    print('Additional Setup ... ', end='')
    if(config['device']['prefer_gpu']):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    model.to(device)
    
    num_epochs = config['training']['num_epochs']
    batch_size = config['training']['batch_size']
    optimizer, scheduler = build_optimizer(model, initial_lr=config["training"]["learning_rate"])
    criterion = nn.MSELoss()
    print('Complete')

    # Training and testing logic
    data_config = {"lat": lat, "lon": lon, "time": time, "sst_all": sst_all,
                   "fig_num": config['training']['fig_num'],
                   "batch_size": batch_size, "device": device,
                   "start_point": config['training']['start_point'],
                   "sensor_num": config['training']['sensor_num'],
                   "sensor_seed": config['training']['sensor_seed'],
                   "sigma": config['training']['sigma'],
                   "inputType": config['training']['input_type']
                   }
    start_time = t.time()
    training_data = train(model, data_config, optimizer, scheduler, criterion, device, method='base', step_size=16, num_epochs=num_epochs)
    end_time = t.time()
    exec_time = end_time - start_time
    print(f'Training Complete in {exec_time:.2f} seconds')

    # Save model and results
    print('Saving Model and getting training patterns... ')
    torch.save(model.state_dict(), config['output']['model_save_path'])
    
    dump_result(config, training_data, exec_time, mode="train")
    print('Complete')


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python main.py config.yaml")
        sys.exit(1)
    config_path = sys.argv[1]  # Get the configuration file path from the command line
    main(config_path)