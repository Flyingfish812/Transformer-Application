# Package import
import numpy as np
import h5py
import torch
import copy
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
def expand(map):
    a,b = map.shape
    size_map = max(a,b)
    size_other = min(a,b)
    res = np.zeros((size_map,size_map))
    range_a = int(size_map/2-size_other/2)
    range_b = int(size_map/2+size_other/2)
    if a < b:
        res[range_a:range_b,:] = map
    else:
        res[:,range_a:range_b] = map
    return res

# Turning a map into an input that can be used by model
# The source is considered as a 180*360 matrix
def to_input(source, device, inputType = "VIT"):
    if(inputType == "VIT"):
        source = expand(source)
    tensor = torch.from_numpy(source.copy())    # To a tensor
    tensor = tensor.to(torch.float32)
    tensor = tensor.unsqueeze(0).unsqueeze(0)   # Fit the size
    if(inputType == "VIT" or inputType == "RES"):
        tensor = tensor.repeat(1,3,1,1)
    tensor = tensor.to(device)                  # To specific device
    return tensor

# If you want to get one sample to visualize
def get_one_sample(n: int, lat, lon, time, sst_all, device, 
                   sigma = 0, sensor_num = 15, sensor_seed = 200, sparse_location = None, inputType = "VIT"):
    sst = sst_all[n]
    sst_obj = SST(lat, lon, sst, time, sensor_num, sensor_seed, sparse_location, sigma)
    target, source, source_map = sst_obj.get_trainer()
    source = to_input(source, device, inputType = inputType)
    source_map = to_input(source_map, device, inputType = inputType)
    target = torch.from_numpy(target.copy()).to(device)
    return target, source, source_map

# Generate the data sets
def data_geneator(lat, lon, time, sst_all, batch_size, device, start_point = 0, sensor_num = None, sensor_seed = None, sigma = 0, inputType = "VIT"):
    if(sensor_num == None):
        sensor_num = [10, 20, 30, 50, 100]
    if(sensor_seed == None):
        sensor_seed = [1, 10, 25, 51, 199]
    for i in range(batch_size):
        for j in sensor_num:
            for k in sensor_seed:
                target, source, source_map = get_one_sample(start_point+i, 
                                                            lat, lon, time, sst_all, 
                                                            device, sigma, 
                                                            sensor_num = j, sensor_seed = k,
                                                            inputType = inputType)
                
                yield (target, source, source_map)

# Train function
from torch.optim.swa_utils import AveragedModel, SWALR
from torch.optim.lr_scheduler import CosineAnnealingLR
def train(model, data_loader, optimizer, criterion, method = 'base', swa_start = 10):
    total_loss = 0.0
    count = 0
    if(method == 'SWA'):
        swa_model = AveragedModel(model)
        scheduler = CosineAnnealingLR(optimizer, T_max=100)
        swa_scheduler = SWALR(optimizer, swa_lr=0.05)

    model.train()
    for target, source, source_map in tqdm(data_loader):
        count += 1
        optimizer.zero_grad()
        output = model(source)
        output = output.view(180,360)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if(method == 'SWA'):
            if count >= swa_start:
                swa_model.update_parameters(model)
                swa_scheduler.step()
            else:
                scheduler.step()
        total_loss += loss.item()
    return total_loss / count

# Test function
def test(model, data_loader, criterion, device, norm = "L2"):
    model.eval()
    error = 0.0
    max_error = 0.0
    min_error = 10000.0
    test_number = 0
    with torch.no_grad():
        for target, source, source_map in tqdm(data_loader):
            # batch_input = source.to(torch.float32).repeat(1,3,1,1).to(device)
            # batch_target = target.to(torch.float32).repeat(1,3,1,1).to(device)
            output = model(source)
            output = output.view(180,360)
            
            # ---- L2 relative loss ----
            if norm == "L2":
                loss = torch.norm(output - target, p=2)
                norm = torch.norm(target, p=2)
                error_val = loss / norm
                error = error + error_val
                if error_val > max_error: max_error = error_val
                if error_val < min_error: min_error = error_val

            # ---- L infinity relative loss ----
            elif norm == "Linf":
                loss = torch.max(torch.abs(output-target))
                norm = torch.max(target)
                error_val = loss / norm
                error = error + error_val
                if error_val > max_error: max_error = error_val
                if error_val < min_error: min_error = error_val

            test_number += 1
    return error, test_number, max_error, min_error

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
        return np.nan_to_num(sst_reshape, nan = 0)
    
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
