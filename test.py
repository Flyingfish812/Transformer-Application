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
    session_num = (config["testing"]["fig_num"]*len(config["training"]["sensor_num"]+config["testing"]["sensor_num"])*
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

    criterion = nn.MSELoss()
    print('Complete')

    # Testing the model
    sensornum_trained = config['training']['sensor_num']
    sensornum_untrained = config['testing']['sensor_num']
    sensorseed_trained = config['training']['sensor_seed']
    sensorseed_untrained = config['testing']['sensor_seed']
    avg_trained = []
    train_trained = []
    test_trained = []
    avg_untrained = []
    train_untrained = []
    test_untrained = []
    for i in range(len(sensornum_trained)):
        data_loader = data_generator(lat, lon, time, sst_all,
                                    fig_num = config['testing']['fig_num'],
                                    batch_size = config['testing']['batch_size'],
                                    device = device, 
                                    start_point = config['testing']['start_point'],
                                    sensor_num = [sensornum_trained[i]],
                                    sensor_seed = sensorseed_trained,
                                    sigma = config['testing']['sigma'],
                                    inputType = config['testing']['input_type'],
                                    onlytest = False)
        avg_loss, _, train_loss, test_loss = test(model, data_loader, device, norm=config['testing']['norm'])
        avg_trained.append(avg_loss)
        train_trained.append(train_loss)
        test_trained.append(test_loss)
        data_loader = data_generator(lat, lon, time, sst_all,
                                    fig_num = config['testing']['fig_num'],
                                    batch_size = config['testing']['batch_size'],
                                    device = device, 
                                    start_point = config['testing']['start_point'],
                                    sensor_num = [sensornum_trained[i]],
                                    sensor_seed = sensorseed_untrained,
                                    sigma = config['testing']['sigma'],
                                    inputType = config['testing']['input_type'],
                                    onlytest = False)
        avg_loss, _, train_loss, test_loss = test(model, data_loader, device, norm=config['testing']['norm'])
        avg_untrained.append(avg_loss)
        train_untrained.append(train_loss)
        test_untrained.append(test_loss)
    for i in range(len(sensornum_untrained)):
        data_loader = data_generator(lat, lon, time, sst_all,
                                    fig_num = config['testing']['fig_num'],
                                    batch_size = config['testing']['batch_size'],
                                    device = device, 
                                    start_point = config['testing']['start_point'],
                                    sensor_num = [sensornum_untrained[i]],
                                    sensor_seed = sensorseed_trained,
                                    sigma = config['testing']['sigma'],
                                    inputType = config['testing']['input_type'],
                                    onlytest = False)
        avg_loss, _, train_loss, test_loss = test(model, data_loader, device, norm=config['testing']['norm'])
        avg_trained.append(avg_loss)
        train_trained.append(train_loss)
        test_trained.append(test_loss)
        data_loader = data_generator(lat, lon, time, sst_all,
                                    fig_num = config['testing']['fig_num'],
                                    batch_size = config['testing']['batch_size'],
                                    device = device, 
                                    start_point = config['testing']['start_point'],
                                    sensor_num = [sensornum_untrained[i]],
                                    sensor_seed = sensorseed_untrained,
                                    sigma = config['testing']['sigma'],
                                    inputType = config['testing']['input_type'],
                                    onlytest = False)
        avg_loss, _, train_loss, test_loss = test(model, data_loader, device, norm=config['testing']['norm'])
        avg_untrained.append(avg_loss)
        train_untrained.append(train_loss)
        test_untrained.append(test_loss)
    
    test_result = {'sensor_num': sensornum_trained+sensornum_untrained, 'avg_trained': avg_trained,'avg_train': train_trained, 'avg_test': test_trained,
                   'avg_untrained': avg_untrained, 'avg_untrained_train': train_untrained, 'avg_untrained_test': test_untrained}
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
