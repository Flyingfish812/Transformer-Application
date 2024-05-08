import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt

'''
The Structured Training Unit (STU) class.
What should be included in the infolist:
    - 'lat': latitude, the first dimension of the input and output
    - 'lon': longitude, the second dimension of the input and output
    - 'nan': the value to replace NaN in the input and output
    - 'sensor_seed': the seed for the random number generator
    - 'sensor_num': the number of sensors
Optional:
    - 'lat_list': the list for the latitude axis
    - 'lon_list': the list for the longitude axis
    - 'map': the locations of the sensors
    - 'need_sparse': whether to generate a sparse map
    - 'sparse_method': the method to generate the sparse map
    - 'sigma': the standard deviation of the Gaussian distribution, used for Gaussian noise
'''
class STU:
    def __init__(self, input, output, infolist):
        self.input = input
        self.output = output
        self.infolist = infolist
    
    def reshape(self, target):
        if not ('lat' in self.infolist and 'lon' in self.infolist):
            raise ValueError("infolist should contain 'lat' and 'lon'")
        target_lat = self.infolist['lat']
        target_lon = self.infolist['lon']
        reshape_output = np.reshape(target, (target_lat, target_lon), order = 'F')
        return reshape_output
    
    def fill_nan(self, target):
        if not ('nan' in self.infolist):
            raise ValueError("infolist should contain 'nan'")
        fill_output = np.nan_to_num(target, nan = self.infolist['nan'])
        return fill_output
    
    def set_sparse_map(self, target, method = 'random'):
        if method == 'random':
            if not ('sensor_seed' in self.infolist and 'sensor_num' in self.infolist):
                raise ValueError("infolist should contain 'sensor_seed' and 'sensor_num'")
            if not ('lat' in self.infolist and 'lon' in self.infolist):
                raise ValueError("infolist should contain 'lat' and 'lon'")
            
            np.random.seed(self.infolist['sensor_seed'])
            sparse_locations = np.zeros((self.infolist['sensor_num'],2))
            for i in range(self.infolist['sensor_num']):
                sparse_locations_lat = np.random.randint(self.infolist['lat'])
                sparse_locations_lon = np.random.randint(self.infolist['lon'])
                # If sensor find a nan value, then select another one
                while target[int(sparse_locations_lat),int(sparse_locations_lon)] == np.nan:
                    sparse_locations_lat = np.random.randint(self.infolist['lat'])
                    sparse_locations_lon = np.random.randint(self.infolist['lon'])
                sparse_locations[i,0] = sparse_locations_lat
                sparse_locations[i,1] = sparse_locations_lon
            
            self.infolist['map'] = sparse_locations
    
    def get_sparse_data(self, target):
        if not ('map' in self.infolist):
            raise ValueError("infolist should contain 'map'")
        if not ('sensor_num' in self.infolist):
            raise ValueError("infolist should contain 'sensor_num'")
        if not ('sigma' in self.infolist):
            sigma = 0.0
        else:
            sigma = self.infolist['sigma']
        
        res = np.zeros((self.infolist['lat'], self.infolist['lon']))
        res_map = np.zeros((self.infolist['lat'], self.infolist['lon']))

        sparse_locations = self.infolist['map']
        sparse_data = np.zeros((self.infolist['sensor_num']))
        for i in range(self.infolist['sensor_num']):
            [locations_lat, locations_lon] = sparse_locations[i]
            noise_factor = np.random.normal(0, sigma)
            sparse_data[i] = (target[int(locations_lat), int(locations_lon)]) * (1 + noise_factor)  # Add a random noise

        lat_list = np.array(list(range(self.infolist['lat'])))
        lon_list = np.array(list(range(self.infolist['lon'])))
        xv1, yv1 =np.meshgrid(lat_list, lon_list)
        grid_data = griddata(sparse_locations, sparse_data, (xv1,yv1), method='nearest')
        res[:,:] = np.transpose(np.nan_to_num(grid_data, nan = 0))

        # Get the mask: if sensor is in this location then note 1 here
        for i in range(self.infolist['sensor_num']):
            [locations_lat, locations_lon] = sparse_locations[i]
            res_map[int(locations_lat),int(locations_lon)] = 1
        
        res = np.flipud(res)
        res_map = np.flipud(res_map)

        return res, res_map
    
    def set_trainer(self):
        need_sparse = self.infolist['need_sparse'] if 'need_sparse' in self.infolist else True
        if need_sparse:
            original_result = np.flipud(self.reshape(self.output))
            real_result = np.flipud(self.fill_nan(self.reshape(self.output)))
            self.set_sparse_map(original_result)
            detect_result, detect_location = self.get_sparse_data(real_result)
            self.input = detect_result
            self.output = real_result
            self.infolist['map_location'] = detect_location
        else:
            self.input = self.reshape(self.output)
            self.output = self.reshape(self.output)
            self.infolist['map_location'] = np.zeros((self.infolist['lat'], self.infolist['lon']))
    
    def display(self, target):
        if target == 'input':
            plt.imshow(self.input)
        elif target == 'output':
            plt.imshow(self.output)
        elif target == 'map':
            if not ('map_location' in self.infolist):
                raise ValueError("You need to have 'map_location' in infolist. To achieve this, please run set_trainer() first.")
            if not ('lat' in self.infolist and 'lon' in self.infolist):
                raise ValueError("infolist should contain 'lat' and 'lon'")
            map_location = self.infolist['map_location']
            y, x = np.where(map_location == 1)
            plt.figure(figsize=(10,5))
            plt.scatter(x, y, color='red') # Use scatter for individual points, better for large data
            plt.xlim(0, self.infolist['lon']) # Set x limits to match array dimensions
            plt.ylim(0, self.infolist['lat']) # Set y limits to match array dimensions
            plt.gca().set_aspect('auto') # Set aspect ratio to 'auto' to fit the array size
            plt.gca().invert_yaxis() # Invert the y-axis to match array indexing
            plt.title('Positions of sensors')
            plt.xlabel('Lontitude')
            plt.ylabel('Latitude')
            plt.show()
        else:
            raise ValueError("target should be 'input', 'output' or 'map'")