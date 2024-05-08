import os
import h5py
import numpy as np
import yaml
import json
import pandas as pd
import datetime

variables = {}

'''
The read function used in this library, we accept the .mat file as an argument and return a dictionary with the data.
'''
def read_mat(file_name:str):
    if not file_name.endswith('.mat'):
        raise ValueError("The file must be a .mat file.")
        
    # Check if the file exists
    if not os.path.isfile(file_name):
        raise FileNotFoundError("The file does not exist.")
    # Open the .mat file in read mode
    with h5py.File(file_name, 'r') as file:
        global variables
        # Iterate over all items in the file, assuming they are datasets
        for name, data in file.items():
            # Convert the dataset to a numpy array and store in dictionary
            variables[name] = np.array(data)
            # Print a message indicating the variable set
            print(f"Variable '{name}' is set.")

    return variables

'''
Read the configuration file.
'''
def read_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

'''
Output function: dump the data into a json file and the result into a markdown file.
'''
def dump_result(config, data_item, exec_time, mode="train"):
    # Step 1: Generate filename based on the current system time
    current_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    if (mode == "train"):
        file_name = f"result/result_train_{current_time}.md"
        data_name = f"result/error_train_{current_time}.json"
    elif (mode == "test"):
        file_name = f"result/result_test_{current_time}.md"
        data_name = f"result/error_test_{current_time}.json"
    else:
        pass

    # Step 2: Dump the training data into json file.
    if (mode == "train"):
        all_data = {}
        for i in range(len(data_item)):
            data = data_item[i]
            # Use the iteration number as the key
            all_data[f'iteration_{i}'] = {
                'train_loss': data[0], 
                'train_error': data[1], 
                'validation_loss': data[2], 
                'validation_error': data[3]
            }
            print(f'train_loss = {data[0][-1]}', end=', ')
            print(f'train_error = {data[1][-1]}')
            print(f'validation_loss = {data[2][-1]}', end=', ')
            print(f'validation_error = {data[3][-1]}')
        with open(data_name, 'w') as file:
            json.dump(all_data, file, indent=4)
    elif (mode == "test"):
        all_data = {}
        i = 1
        for sensor_info, test_error in data_item.items():
            all_data[f'test_error_{i}'] = {
                'sensor_num': sensor_info[0],
                'sensor_seed': sensor_info[1],
                'test_error': test_error
            }
            i += 1
        with open(data_name, 'w') as file:
            json.dump(all_data, file, indent=4)

    
    with open(file_name, 'w') as md_file:
        # Step 3: Write config information to the markdown file
        md_file.write("## Description\n")
        md_file.write(f"- time: {current_time}\n\n")
        for category, attributes in config.items():
            md_file.write(f"## {category}\n")
            for attr, value in attributes.items():
                md_file.write(f"- {attr}: {value}\n")
            md_file.write("\n")  # Add an extra newline for better readability
        
        # Step 4: Write training data to the file
        if mode == 'train':
            md_file.write("## Training Data\n")
            md_file.write(f"Training time: {exec_time:.2f} seconds\n")
            for epoch, metrics in data_item.items():
                # Assuming the structure is like [train_loss, train_error, validation_loss, validation_error]
                train_loss, train_error, validation_loss, validation_error = [metrics_list[-1] for metrics_list in metrics]
                md_file.write(f"Epoch {epoch}:\n train_loss={train_loss}\n train_error={train_error}\n validation_loss={validation_loss}\n validation_error={validation_error}\n")
        elif mode == 'test':
            md_file.write("## Testing Data\n")
            md_file.write(f"Testing time: {exec_time:.2f} seconds\n")
            for sensor_info, test_error in data_item.items():
                # Assuming the structure is like test_error
                md_file.write(f"Number of sensor: {sensor_info[0]}; Seed for sensor: {sensor_info[1]}\n")
                md_file.write(f"Testing error: {sum(test_error) / len(test_error)}\n")