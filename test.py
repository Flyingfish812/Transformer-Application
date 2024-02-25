from presets import *
from my_utils import *
import sys
import torch
import torch.nn as nn
import torch.optim as optim

def main(config_path):
    print('----- Testing Session -----')
    # Read config
    print('Reading Config ... ', end='')
    config = read_config(config_path)
    print('Complete')
    print(f'Model Name: {config["model"]["name"]}')
    print(f'Expected sessions: {len(config["testing"]["sensor_num"])*(len(config["testing"]["sensor_seed"]+config["training"]["sensor_num"]))*config["testing"]["batch_size"]}')

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

    criterion = nn.MSELoss()
    print('Complete')

    # Testing the model
    sensornum = config['testing']['sensor_num']
    sensorseed_trained = config['training']['sensor_seed']
    sensorseed_untrained = config['testing']['sensor_seed']
    result_trained = []
    max_trained = []
    min_trained = []
    result_untrained = []
    max_untrained = []
    min_untrained = []
    for i in range(len(sensornum)):
        data_loader = data_generator(lat, lon, time, sst_all,
                                    batch_size = config['testing']['batch_size'],
                                    device = device, 
                                    start_point = config['testing']['start_point'],
                                    sensor_num = [sensornum[i]],
                                    sensor_seed = sensorseed_trained,
                                    sigma = config['testing']['sigma'],
                                    inputType = config['testing']['input_type'],
                                    onlytest = True)
        avg_loss, _, max_loss, min_loss = test(model, data_loader, device, norm=config['testing']['norm'])
        result_trained.append(avg_loss)
        max_trained.append(max_loss)
        min_trained.append(min_loss)
    for i in range(len(sensornum)):
        data_loader = data_generator(lat, lon, time, sst_all,
                                    batch_size = config['testing']['batch_size'],
                                    device = device, 
                                    start_point = config['testing']['start_point'],
                                    sensor_num = [sensornum[i]],
                                    sensor_seed = sensorseed_untrained,
                                    sigma = config['testing']['sigma'],
                                    inputType = config['testing']['input_type'],
                                    onlytest = True)
        avg_loss, _, max_loss, min_loss = test(model, data_loader, device, norm=config['testing']['norm'])
        result_untrained.append(avg_loss)
        max_untrained.append(max_loss)
        min_untrained.append(min_loss)
    
    test_result = {'sensor_num': sensornum, 'trained_results': result_trained,'trained_max': max_trained, 'trained_min': min_trained,
                   'untrained_results': result_untrained, 'untrained_max': max_untrained, 'untrained_min': min_untrained}
    print('Testing Complete')

    # Save the results
    print('Saving Result ... ', end='')
    save_results(config_path, test_result, output_type = 'xlsx', results_output = config['output']['model_result'])
    print('Complete')


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python test.py config.yaml")
        sys.exit(1)
    config_path = sys.argv[1]  # Get the configuration file path from the command line
    main(config_path)
