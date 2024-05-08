# Package import
import numpy as np
import h5py
import torch
import copy
import torch.nn as nn
from tqdm import tqdm
from scipy.interpolate import griddata

# Example: file_name = './sst_weekly.mat'
def read_hdf5(file_name):
    file = h5py.File(file_name,'r')
    lat = np.array(file['lat'])
    lon = np.array(file['lon'])
    sst_all = np.array(file['sst'])
    time = np.array(file['time'])
    return (lat,lon,sst_all,time)

# The input is a 180*360 map, it's used to expand to 360*360
# Another method applied: mirroring
def expand(map):
    map_mirror = np.flipud(map)
    res = np.vstack((map,map_mirror))
    return res

# Turning a map into an input that can be used by model
# The source is considered as a 180*360 matrix
def to_input(source, device, inputType = "VIT"):
    if(inputType == "VIT"):
        source = expand(source)
    tensor = torch.from_numpy(source.copy())    # To a tensor
    tensor = tensor.to(torch.float32)
    # tensor = tensor.unsqueeze(0).unsqueeze(0)   # Fit the size
    # if(inputType == "VIT" or inputType == "RES"):
    #     tensor = tensor.repeat(1,3,1,1)
    tensor = tensor.to(device)                  # To specific device
    return tensor

# If you want to get one sample to visualize
def get_one_sample(n: int, lat, lon, time, sst_all, device, 
                   sigma = 0, sensor_num = 15, sensor_seed = 200, sparse_location = None, inputType = "VIT"):
    n = int(n)
    sst = sst_all[n]
    sst_obj = SST(lat, lon, sst, time, sensor_num, sensor_seed, sparse_location, sigma)
    target, source, source_map = sst_obj.get_trainer()
    source = to_input(source, device, inputType = inputType)
    source_map = to_input(source_map, device, inputType = inputType)
    target = torch.from_numpy(target.copy()).to(device).unsqueeze(0)    
    # if inputType == "CNN":
    #     combined_source = source.unsqueeze(0).unsqueeze(0)
    # else:
    #     combined_source = torch.stack([source.unsqueeze(0), source_map.unsqueeze(0), source.unsqueeze(0) * source_map.unsqueeze(0)], dim=1)
    combined_source = torch.stack([source.unsqueeze(0), source_map.unsqueeze(0), source.unsqueeze(0) * source_map.unsqueeze(0)], dim=1)
    return target, combined_source

# Function for cross validation, use swap to ensure that each 10% data can be the validation set.
def n_swap(arr, n):
    num_elements = len(arr)
    swap_size = int(num_elements * n * 0.1)
    last_part = arr[-swap_size:]
    first_part = arr[:-swap_size]
    return np.concatenate([last_part, first_part])

# Generate the data sets
def data_generator(lat, lon, time, sst_all, fig_num, batch_size, device,
                   start_point=0, sensor_num=None, sensor_seed=None,
                   sigma=0, inputType="VIT", n=0, mask = None, onlytest=False):
    if sensor_num is None:
        sensor_num = [10, 20, 30, 50, 100]
    if sensor_seed is None:
        sensor_seed = [1, 10, 25, 51, 199]

    total_samples = fig_num * len(sensor_num) * len(sensor_seed) // batch_size
    train_cutoff = int(0.9 * total_samples)
    verify_cutoff = train_cutoff + int(0.1 * total_samples)

    current_sample = 0
    targets, sources = [], []
    fig_num_list = list(range(fig_num))
    np.random.seed(0)
    np.random.shuffle(fig_num_list)
    fig_num_list = n_swap(fig_num_list, n)

    for j in sensor_num:
        for k in sensor_seed:
            for i in fig_num_list:
                target, source = get_one_sample(start_point + i,
                                                lat, lon, time, sst_all,
                                                device, sigma,
                                                sensor_num=j, sensor_seed=k,
                                                inputType=inputType)
                if onlytest:
                    category = 'test'
                elif current_sample < train_cutoff:
                    category = 'train'
                elif current_sample < verify_cutoff:
                    category = 'verify'
                else:
                    category = 'test'

                if mask is not None:
                    source = source * mask
                targets.append(target)
                sources.append(source)

                # Check if the batch has reached the specified batch size
                if len(targets) == batch_size:
                    yield (torch.cat(targets, dim=0), torch.cat(sources, dim=0), category)
                    current_sample += 1
                    targets, sources = [], []

            # After the loop, yield any remaining samples in the batch
            if targets:
                yield (torch.cat(targets, dim=0), torch.cat(sources, dim=0), category)
                current_sample += 1
                targets, sources = [], []

def factory(data_config, onlytest=False):
    lat, lon, time, sst_all = data_config['lat'], data_config['lon'], data_config['time'], data_config['sst_all']
    fig_num = data_config['fig_num']
    batch_size = data_config['batch_size']
    device = data_config['device']
    start_point = data_config['start_point']
    sensor_num = data_config['sensor_num']
    sensor_seed = data_config['sensor_seed']
    sigma = data_config['sigma']
    inputType = data_config['inputType']
    if 'n' in data_config:
        n = data_config['n']
    else:
        n = 0
    # mask = data_config['mask']
    data_gen = data_generator(lat, lon, time, sst_all, fig_num, batch_size, device, start_point, sensor_num, sensor_seed, sigma, inputType, n=n, onlytest=onlytest)
    return data_gen

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

# Train function
from torch.optim.swa_utils import AveragedModel, SWALR
def train(model, data_config, optimizer, scheduler, criterion, device, method='base', step_size=16, num_epochs=2):
    training_data = {}
    total_loss = 0.0  # Used when using "base" method
    count = 0
    validation_count = 0

    if method == 'SWA':
        swa_model = AveragedModel(model)
        swa_scheduler = SWALR(optimizer, swa_lr=0.05, anneal_strategy='cos')

    for epoch in range(num_epochs):
        model.train()
        train_loss, train_error = [], []
        validation_loss, validation_error = [], []
        data_gen = factory(data_config)
        # if epoch > 0:
        #     for param_group in optimizer.param_groups:
        #         param_group['lr'] *= 0.1
        for target, source, category in tqdm(data_gen):
            if category == 'train':
                count += 1
                optimizer.zero_grad()
                output = model(source.to(device))
                output = output.view(-1, 180, 360)
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
                    output = output.view(-1, 180, 360)
                    loss = criterion(output, target.to(device))
                    error = calculate_norm(output, target.to(device))
                    validation_loss.append(loss.item())
                    validation_error.append(error)
                # model.train()

        if method == 'SWA':
            swa_model.update_parameters(model)
        training_data[epoch] = [train_loss, train_error, validation_loss, validation_error]
    
    return training_data

# Rewritten test function
def test(model, data_config, device, norm="L2"):
    testing_data = {}
    sensor_num_list = data_config['sensor_num']
    sensor_seed_list = data_config['sensor_seed']
    model.eval()

    with torch.no_grad():
        for sensor_num in sensor_num_list:
            for sensor_seed in sensor_seed_list:
                data_config['sensor_num'] = [sensor_num]
                data_config['sensor_seed'] = [sensor_seed]
                test_error = []
                data_gen = factory(data_config, onlytest=True)
                for target, source, _ in tqdm(data_gen):
                    output = model(source.to(device))
                    output = output.view(-1, 180, 360)
                    error = calculate_norm(output, target.to(device), type=norm)
                    test_error.append(error)
                avg_error = sum(test_error) / len(test_error)
                print(f'With {sensor_num} sensors at seed {sensor_seed} get an average {norm} error of {avg_error}')
                testing_data[(sensor_num, sensor_seed)] = test_error
        return testing_data

# An SST object contains a sample map and an overall map
class SST:
    # Latitude list, Longtitude list, one sst value table, measured time, number of sensor, seed for distribution
    def __init__(self, lat, lon, sst, time, sensor_num, sensor_seed, sparse_location = None, sigma = 0):
        self.lat = lat[0]
        self.lon = lon[0]
        self.sst = sst
        self.time = time
        self.sensor_num = sensor_num
        self.sensor_seed = sensor_seed
        self.sparse_location = sparse_location
        self.sigma = sigma  # Coefficient of Gaussian noise

        self.lat_length = len(self.lat)
        self.lon_length = len(self.lon)
        
    def sst_reshape(self):
        sst_reshape = np.reshape(self.sst,(self.lat_length,self.lon_length),order = 'F')
        return sst_reshape
    
    def fill_nan(self,sst_reshape):
        return np.nan_to_num(sst_reshape, nan = -273)
    
    def get_real_value(self):
        return self.fill_nan(self.sst)

    # Get the map detected by sensors
    def get_mask_img(self):
        sst_map = self.sst_reshape()  # With nan value
        sst_data = self.fill_nan(sst_map)  # Fill all the nan with 0
        res = np.zeros((self.lat_length,self.lon_length))
        res_map = np.zeros((self.lat_length,self.lon_length))
        
        # Choose the sensors' location
        if(self.sparse_location == None):
            np.random.seed(self.sensor_seed)
            sparse_locations = np.zeros((self.sensor_num,2))
            for i in range(self.sensor_num):
                sparse_locations_lat = np.random.randint(self.lat_length)
                sparse_locations_lon = np.random.randint(self.lon_length)
                # If sensor find a nan value, then select another one
                while sst_map[int(sparse_locations_lat),int(sparse_locations_lon)] == np.nan:
                    sparse_locations_lat = np.random.randint(self.lat_length)
                    sparse_locations_lon = np.random.randint(self.lon_length)
                sparse_locations[i,0] = sparse_locations_lat
                sparse_locations[i,1] = sparse_locations_lon
        else:
            sparse_locations = self.sparse_location
        
        # Get the sample value
        sparse_data = np.zeros((self.sensor_num))
        for i in range(self.sensor_num):
            [locations_lat, locations_lon] = sparse_locations[i]
            noise_factor = np.random.normal(0,self.sigma)
            sparse_data[i] = (sst_data[int(locations_lat),int(locations_lon)]) * (1 + noise_factor)  # Add a random noise
        
        # Get the exact location of sample
        sparse_locations_exact = np.zeros(sparse_locations.shape)
        for i in range(self.sensor_num):
            [locations_lat, locations_lon] = sparse_locations[i]
            sparse_locations_exact[i,0] = self.lat[int(locations_lat)]
            sparse_locations_exact[i,1] = self.lon[int(locations_lon)]

        # Interpolation
        xv1, yv1 =np.meshgrid(self.lat, self.lon)
        grid_data = griddata(sparse_locations_exact, sparse_data, (xv1,yv1), method='nearest')
        res[:,:] = np.transpose(np.nan_to_num(grid_data, nan = 0))

        # Get the mask: if sensor is in this location then note 1 here
        for i in range(self.sensor_num):
            [locations_lat, locations_lon] = sparse_locations[i]
            res_map[int(locations_lat),int(locations_lon)] = 1
        
        res = np.flipud(res)
        res_map = np.flipud(res_map)

        return res, res_map
    
    def get_trainer(self):
        real_result = np.flipud(self.fill_nan(self.sst_reshape()))
        detect_result, detect_map = self.get_mask_img()
        return (real_result, detect_result, detect_map)
