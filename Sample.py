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
def to_input(source, device, doexpand = False):
    if doexpand:
        source = expand(source)
    return torch.from_numpy(source.copy()).unsqueeze(0).unsqueeze(0).to(torch.float32).repeat(1,3,1,1).to(device)

# If you want to get one sample to visualize
def get_one_sample(n: int, lat, lon, time, sst_all, device, 
                   sensor_num = 15, sensor_seed = 200, sparse_location = None, toinput = True):
    sst = sst_all[n]
    sst_obj = SST(lat, lon, sst, time, sensor_num, sensor_seed, sparse_location)
    target, source, source_map = sst_obj.get_trainer()
    if toinput:
        source = to_input(source, device, True)
        source_map = to_input(source_map, device, True)
        target = to_input(target, device, False)
    else:
        source = source
        target = target
    return target, source, source_map

# Generate the data sets
def data_geneator(lat, lon, time, sst_all, batch_size, device, start_point = 0, sensor_num = None, sensor_seed = None):
    if(sensor_num == None):
        sensor_num = [10, 20, 30, 50, 100]
    if(sensor_seed == None):
        sensor_seed = [1, 10, 25, 51, 199]
    for i in range(batch_size):
        for j in sensor_num:
            for k in sensor_seed:
                target, source, source_map = get_one_sample(start_point+i, lat, lon, time, sst_all, device, j, k)
                # sst = sst_all[start_point+i]
                # sst_obj = SST(lat, lon, sst, time, j, k)
                # target, source, source_map = sst_obj.get_trainer()
                # target = torch.from_numpy(target.copy()).unsqueeze(0).unsqueeze(0)
                # source = torch.from_numpy(expand(source).copy()).unsqueeze(0).unsqueeze(0)
                # source_map = torch.from_numpy(expand(source_map).copy()).unsqueeze(0).unsqueeze(0)
                yield (target, source, source_map)

# Train function
def train(model, data_loader, optimizer, criterion, device):
    total_loss = 0.0
    model.train()
    for target, source, source_map in tqdm(data_loader):
        # batch_input = source.to(torch.float32).repeat(1,3,1,1).to(device)
        # batch_target = target.to(torch.float32).repeat(1,3,1,1).to(device)
        optimizer.zero_grad()
        output = model(source)
        #output = output.view(360,360)
        output = output.view(180,360)
        loss = criterion(output, target[0][0])
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss

# Test function
def test(model, data_loader, criterion, device):
    model.eval()
    error = 0.0
    test_number = 0
    with torch.no_grad():
        for target, source, source_map in tqdm(data_loader):
            # batch_input = source.to(torch.float32).repeat(1,3,1,1).to(device)
            # batch_target = target.to(torch.float32).repeat(1,3,1,1).to(device)
            output = model(source)
            #output = output.view(360,360)
            output = output.view(180,360)
            # L2 relative loss
            # loss = criterion(output, target[0][0])
            # norm = torch.norm(target) ** 2
            # error += loss.item() / norm * 180 * 360

            # L infinity relative loss 
            loss = torch.max(torch.abs(output-target[0][0]))
            norm = torch.max(target[0][0])
            error = loss / norm
            test_number += 1
    return error, test_number

# An SST object contains a sample map and an overall map
class SST:
    # Latitude list, Longtitude list, one sst value table, measured time, number of sensor, seed for distribution
    def __init__(self, lat, lon, sst, time, sensor_num, sensor_seed, sparse_location = None):
        self.lat = lat[0]
        self.lon = lon[0]
        self.sst = sst
        self.time = time
        self.sensor_num = sensor_num
        self.sensor_seed = sensor_seed
        self.sparse_location = sparse_location

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
            sparse_data[i] = (sst_data[int(locations_lat),int(locations_lon)])
        
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
