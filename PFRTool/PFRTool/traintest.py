import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from .iofunctions import variables
from .torchfunctions import *
from torch.optim.swa_utils import AveragedModel, SWALR

'''
This is the train function, it takes 2 lists other than the model itself: setup and infolist.
What should be included in the setup:
    - optimizer
    - scheduler
    - criterion
    - device
    - method
    - train_epoch
    - step_size
You can finalize the setup by calling the function "setup_info()" in the "torchfunctions" file.
    
What should be included in the infolist:
    - lat
    - lon
'''
def train(model, setup, infolist):
    global variables
    optimizer = setup['optimizer']
    scheduler = setup['scheduler']
    criterion = setup['criterion']
    device = setup['device']
    method = setup['method']
    epochs = setup['train_epoch']
    step_size = setup['step_size']
    output_lat = infolist['lat']
    output_lon = infolist['lon']

    training_data = {}
    total_loss = 0.0  # Used when using "base" method
    count = 0
    validation_count = 0

    if method == 'SWA':
        swa_model = AveragedModel(model)
        swa_scheduler = SWALR(optimizer, swa_lr=0.05, anneal_strategy='cos')
    
    for epoch in range(epochs):
        model.train()
        train_loss, train_error = [], []
        validation_loss, validation_error = [], []
        data_gen = factory(variables, setup, infolist)
        # if epoch > 0:
        #     for param_group in optimizer.param_groups:
        #         param_group['lr'] *= 0.1
        for target, source, category in tqdm(data_gen):
            if category == 'train':
                count += 1
                optimizer.zero_grad()
                output = model(source.to(device))
                output = output.view(-1, output_lat, output_lon)
                loss = criterion(output, target.to(device))
                error = calculate_norm(output, target.to(device))
                loss.backward()
                optimizer.step()
                train_loss.append(loss.item())
                train_error.append(error)
                total_loss += loss.item()

                # Adjust learning rate for basic training method after every step_size batches
                if method == 'base' and count % step_size == 0:
                    avg_loss = total_loss / step_size
                    scheduler.step(avg_loss)  # Adjust learning rate based on the average loss of recent batches
                    total_loss = 0.0  # Reset total loss for the next set of batches

                # For SWA, update parameters and adjust SWA scheduler after step_size batches
                elif method == 'SWA' and count - step_size >= 0:
                    swa_model.update_parameters(model)
                    swa_scheduler.step()

            elif category == 'verify':
                model.eval()
                with torch.no_grad():
                    validation_count += 1
                    output = model(source.to(device))
                    output = output.view(-1, output_lat, output_lon)
                    loss = criterion(output, target.to(device))
                    error = calculate_norm(output, target.to(device))
                    validation_loss.append(loss.item())
                    validation_error.append(error)
                # model.train()

        if method == 'SWA':
            swa_model.update_parameters(model)
        training_data[epoch] = [train_loss, train_error, validation_loss, validation_error]
    
    return training_data

def test(model, setup, infolist, norm="L2"):
    global variables
    device = setup['device']
    testing_data = {}
    sensor_num_list = infolist['sensor_num']
    sensor_seed_list = infolist['sensor_seed']
    model.eval()

    with torch.no_grad():
        for sensor_num in sensor_num_list:
            for sensor_seed in sensor_seed_list:
                infolist['sensor_num'] = sensor_num
                infolist['sensor_seed'] = sensor_seed
                test_error = []
                data_gen = factory(variables, setup, infolist)
                for target, source, _ in tqdm(data_gen):
                    output = model(source.to(device))
                    output = output.view(-1, 180, 360)
                    error = calculate_norm(output, target.to(device), type=norm)
                    test_error.append(error)
                avg_error = sum(test_error) / len(test_error)
                print(f'With {sensor_num} sensors at seed {sensor_seed} get an average {norm} error of {avg_error}')
                testing_data[(sensor_num, sensor_seed)] = test_error
        return testing_data
    