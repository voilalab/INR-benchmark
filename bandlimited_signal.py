import numpy as np
import scipy
import matplotlib.pyplot as plt
import numpy.fft as fft

import sys
sys.path.append('../')
import random
import torch
import os
from tqdm import tqdm
import time
import datetime
import pytz
import csv
import cv2
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

def bandlimit_filter(data, cutoff_low, cutoff_high):
    """
    bandlimiting filter with "circular" mask
    
    Args:
        data (np.ndarray): input data to be filtered
        cutoff_low (float): low frequency cutoff region (0 ~ 0.5, 0.5 is Nyquist freq.).
        cutoff_high (float): high frequency cutoff region (0 ~ 0.5, 0. is Nyquist freq.).
        
    Returns:
        np.ndarray: Bandlimited Data
    """
    # Discrete Fourier Transform 
    data_fft = fft.fftn(data)
    
    # make meshgrid in the frequency space
    frequencies = [fft.fftfreq(n, d=1.0) for n in data.shape]
    grid = np.meshgrid(*frequencies, indexing='ij')
    radius = np.sqrt(np.sum(np.array(grid) ** 2, axis=0))

    # Generate bandlimit mask
    mask = (cutoff_low/np.sqrt(2) <= radius) & (radius <= cutoff_high/np.sqrt(2))
    
    # apply filter using generated mask
    data_fft_filtered = data_fft * mask
    # vis_filter = np.log(np.abs(fft.fftshift(data_fft_filtered)))
    
    # reconstruct signal using ifft
    filtered_data = np.real(fft.ifftn(data_fft_filtered))
    
    return filtered_data
def generate_bandlimits(start, stop, num_points, base):
    """
    Generate non-linear spaced bandlimits.
    
    Args:
        start (float): Starting value of the bandlimit (e.g., 0.1).
        stop (float): Ending value of the bandlimit (e.g., 0.9).
        num_points (int): Number of bandlimits to generate.
        scale (str): 'log' for logarithmic spacing, 'exp' for exponential spacing.
    
    Returns:
        np.ndarray: Array of bandlimits.
    """
    
    
    # bandlimits = np.linspace(0, 1, num_points) ** 2 * (stop - start) + start
      # Adjust base for more or less steepness
    bandlimits = (np.logspace(0, 1, num_points, base=base) - 1) / (base - 1)
    bandlimits = bandlimits * (stop - start) + start

    return bandlimits


class BandlimitedSignal:
    """
    This signal class generates white noise and then filters it to have a desired maximum spatial frequency.
    """
    def __init__(self, dimension, length, bandlimit, seed= None, generate = True, super_resolution = False, sparse=False):
        self.class_name = self.__class__.__name__
        if generate:
            np.random.seed(seed)
            self.dimension = dimension
            # Make the length odd so that we can have symmetry around the zero frequency
            if length // 2 == length / 2:
                length = length + 1
            self.length = length  # per dimension length; assume same length signal in each dimension
            self.bandlimit = bandlimit  # as a fraction

            # Generate the signal as white noise
            dims = [self.length] * self.dimension
            self.signal = np.random.uniform(size=dims)
            # plt.figure()
            # plt.imshow(self.signal)
            # plt.savefig('signal.png')
            bandlimits = generate_bandlimits(0.0015, 0.7, 9, base = 300)
            # Filter the signal to have the desired bandwidth
            bandlimit_idx = int(str(bandlimit)[-1]) - 1
            bandlimit = bandlimits[bandlimit_idx]
            self.signal = bandlimit_filter(self.signal, 0, bandlimit)
            
            # plt.figure()
            # plt.imshow(self.signal)
            # plt.savefig('filtered_signal.png')
            # plt.figure()
            # plt.imshow(np.abs(filtered_dft))
            # plt.savefig('dft.png')
        else:
            self.signal = np.load(f'target_signals/{self.class_name}/{seed}/{dimension}d_{self.class_name}_{bandlimit}_{seed}.npy')


class SparseSphereSignal:
    """
    This signal class generates random spheres of roughly constant total volume, with a desired radius analogous to bandwidth.
    Please note that generating 3D sphere signal takes a lot of time, I recommend you to use the generated signals.
    """
    def __init__(self, dimension, length, bandlimit, seed, occupied_fraction=0.1, generate = True, super_resolution = False, sparse=False):
        self.class_name = self.__class__.__name__
        if generate:
            np.random.seed(seed)
            # Initialize an empty signal
            self.length = length
            self.bandlimit = bandlimit
            self.dimension = dimension
            dims = [self.length] * self.dimension
            self.signal = np.zeros(dims)

            # Calculate sphere number and radius for this bandwidth
            occupied_cells = occupied_fraction * np.prod(dims)
            self.sphere_radius = self.length / (self.bandlimit * 100)
            sphere_volume = np.pi**(self.dimension / 2.0) * self.sphere_radius**self.dimension / scipy.special.gamma(1 + self.dimension / 2.0)
            self.num_spheres = int(occupied_cells / sphere_volume)

            # Generate spheres
            centers = np.random.uniform(low=0, high=self.length, size=(self.num_spheres, self.dimension))
            it = np.nditer(self.signal, flags=['multi_index'])
            for _ in tqdm(it):
                idx = it.multi_index
                # check if this cell is within the radius of any of the sphere centers
                for center in centers:
                    if np.linalg.norm(idx - center) <= self.sphere_radius:
                        self.signal[idx] = 1
                        break  # If this cell is inside any one sphere, we don't need to bother checking the other spheres
        else:
            self.signal = np.load(f'target_signals/{self.class_name}/{seed}/{dimension}d_{self.class_name}_{bandlimit}_{seed}.npy')


class Sierpinski:
    def __init__(self, dimension, length, bandlimit, seed, generate = True, super_resolution=False, sparse=False):
        self.target_dir = 'target_signals/sierpinski_triangle/1000'
        self.depth = int(bandlimit*10)-1 # changing bandlimit to the depth of Sierpinski Triangle

        if generate:
            # Generate Sierpinski triangle for given depths    
            fig = self.create_sierpinski_plot(self.depth)
            sierpinski_array = self.plot_to_numpy(fig)
            # print(sierpinski_array.shape)
            self.signal = self.post_process(sierpinski_array)
        else:
            self.signal = np.load(f'{self.target_dir}/sierpinski_triangle_1000_depth_{self.depth}.npy').astype('float32')

    # Sierpinski triangle maker
    def sierpinski_triangle(self, vertices, depth, ax):
        if depth == 0:
            # draw triangle
            triangle = plt.Polygon(vertices, edgecolor='black', facecolor='black')
            ax.add_patch(triangle)
        else:
            # calculate the mid points
            midpoints = [(vertices[i] + vertices[(i + 1) % 3]) / 2 for i in range(3)]
            
            # recursively generate small triangles
            self.sierpinski_triangle([vertices[0], midpoints[0], midpoints[2]], depth - 1, ax)
            self.sierpinski_triangle([vertices[1], midpoints[0], midpoints[1]], depth - 1, ax)
            self.sierpinski_triangle([vertices[2], midpoints[1], midpoints[2]], depth - 1, ax)

    # wrapper with fixed 1000x1000 canvas size
    def create_sierpinski_plot(self, depth):
        fig, ax = plt.subplots(figsize=(12, 12), dpi=100)  # figsize=(10, 10) with dpi=100 gives 1000x1000 pixels
        ax.set_aspect('equal')
        
        # initialize vertices for an equilateral triangle
        vertices = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3) / 2]])
        
        # Sierpinski generating function
        self.sierpinski_triangle(vertices, depth, ax)
        
        # disable axis
        ax.set_axis_off()
        
        # set figure boundaries (ensure the triangle fits within the figure)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, np.sqrt(3) / 2)
        
        # Return the figure for further processing
        return fig

    # from Matplotlib to NumPy
    def plot_to_numpy(self, fig):
        # Draw figure on the Canvas
        canvas = FigureCanvas(fig)
        canvas.draw()

        # get RGB data from the canvas
        width, height = fig.get_size_inches() * fig.get_dpi()
        print(f'{width} {height}')
        image = np.frombuffer(canvas.buffer_rgba(), dtype='uint8')
        image = image.reshape(int(height), int(width), 4)

        return image

    # # Post-process the image (convert to binary)
    def post_process(self, data):
        data = data[...,:3]
        data = data.mean(axis = -1)
        H, W = data.shape
        
        Woffset = 15
        data = data[100:1100,100+Woffset:1100+Woffset]
        data = data > 150
        data = 1 - data
        return data


class StarTarget:
    def __init__(self, dimension, length, bandlimit, seed, generate = True, super_resolution=False, num_triangles = 40, sparse=False):
        self.target_dir = 'target_signals/'
        print(f'star_resolution_target_{num_triangles}_1000')
        if generate:
            self.signal, _ = self.star_resolution_target(num_triangles, length)
        else:
            self.signal = np.load(f'{self.target_dir}/star_resolution_target_{num_triangles}_1000.npy').astype('float32')

    # star resolution target generator
    def star_resolution_target(self, num_triangles=40, img_size=1000, dst_edge=20):
        num_triangles *= 2
        center = img_size // 2
        radius = center - dst_edge

        angles = np.linspace(0, 2 * np.pi, num_triangles, endpoint=False)

        palette = np.zeros((img_size, img_size))

        # Use vector-based approach to avoid issues with tan(90 degrees)
        for i, angle in enumerate(angles):
            if i % 2 == 1:
                continue  # Skip every second triangle to create alternating pattern
            else:
                # Define the vectors for two triangle edges
                x1, y1 = radius * np.cos(angles[i]), radius * np.sin(angles[i])
                x2, y2 = radius * np.cos(angles[i + 1]), radius * np.sin(angles[i + 1])

                for x in range(img_size):
                    for y in range(img_size):
                        dx, dy = x - center, y - center
                        dist_from_center = np.sqrt(dx ** 2 + dy ** 2)

                        # Skip points outside the radius
                        if dist_from_center > radius:
                            continue

                        # Use cross products to determine if point is inside the triangle
                        cross1 = np.sign(dx * y1 - dy * x1)
                        cross2 = np.sign(dx * y2 - dy * x2)

                        if cross1 >= 0 and cross2 <= 0:
                            palette[y, x] = 1

        # plt.imshow(palette, cmap='gray')
        # plt.show()
        return palette, radius

class RealImage:
    def __init__(self, dimension, length, bandlimit, seed=0, super_resolution =False, sparse=False):
        target_dir = 'target_signals/DIV2K'
        img_list = ['0064', '0007', '0010', '0029', '0063', '0072', '0079', '0088', '0093', '0131']
        if seed == 1234:
            seed = 0
        elif seed > 1234:
            seed = 0
        if super_resolution:
            self.target_dir = f'{target_dir}/{img_list[seed]}.png'
        else:
            self.target_dir = f'{target_dir}/{img_list[seed]}x4.png'
        
        img = cv2.imread(self.target_dir)
        print(f'dir:{self.target_dir} with seed {seed}')
        self.signal = np.array(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        self.signal = self.signal / 255.0
        tqdm.write(f'{self.signal.shape}')
        
class Voxel_Fitting:
    def __init__(self, dimension, length, bandlimit, seed, super_resolution = False, sparse = True):
        if not sparse:
            if not super_resolution:
                self.dir = 'target_signals/dragon.npy'
            else: self.dir = 'target_signals/dragon_sr.npy'
        else:
            if not super_resolution:
                self.dir = 'target_signals/dragon_sparse.npy'
            else: self.dir = 'target_signals/dragon_sr_sparse.npy'
        self.signal = np.array(np.load(self.dir), dtype = np.int32)
        # tqdm.write(f'{self.dir} {self.signal.shape} {self.signal.shape[0]*self.signal.shape[1]*self.signal.shape[2]}')

def set_seed(seed):
    tqdm.write(f"Selected Seed: {seed}")
    # Python
    random.seed(seed)
    # NumPy
    np.random.seed(seed)
    # PyTorch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)  # if CUDA

if __name__ == '__main__':
    # signal = BandlimitedSignal(2, 100, 0.3)
    class_list = [SparseSphereSignal]
    dimensions = [2, 3]
    target_dir = 'target_signals/'
    # seed_list = [1234, 2024, 5678, 7890, 7618]
    seed_list = [8361]
    

    for signal_class in class_list:
        signal_name = signal_class.__name__
        if not os.path.exists(target_dir+f'{signal_name}'):
            os.makedirs(target_dir+f'{signal_name}')
        for dimension in dimensions:
            if signal_name == "SparseSphereSignal" and dimension == 2:
                tqdm.write(f"{signal_name} {dimension}")
                # continue
            
            for seed in seed_list:
                if not os.path.exists(target_dir+f'{signal_name}/{seed}'):
                    os.makedirs(target_dir+f'{signal_name}/{seed}')
                fieldnames = ['seed', 'bandlimits', 'time']
                if not os.path.exists(target_dir+f'{signal_name}/{seed}'):
                    with open(target_dir+f'{signal_name}/{seed}/estimated_time_{seed}.csv', 'w', newline='') as f:
                        
                        writer = csv.DictWriter(f, fieldnames=fieldnames)
                        writer.writeheader()
                set_seed(seed)
                # bandlimits = [0.1, 0.2, 0.3]

                bandlimits = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
                for bandlimit in tqdm(bandlimits, desc=f'{signal_name} {dimension}'):
                    if dimension == 3:
                        signal_length = 100
                        # continue
                    else:
                        signal_length = 1000
                    tqdm.write(f"{signal_name} {dimension} {bandlimit}")
                    init_time = time.time()
                    signal = signal_class(dimension=dimension, length=signal_length, bandlimit=bandlimit, seed = seed, generate = True)
                    now = datetime.datetime.now(ATL_time)
                    time_spent = time.time() - init_time
                    tqdm.write(f'{signal_name} {dimension} {bandlimit} time: {time_spent:.2f}, now {now.strftime("%Y%m%d_%H%M%S")}')
                    if dimension == 2:
                        plt.figure()
                        plt.imshow(signal.signal)
                        plt.savefig(f'{target_dir}{signal_name}/{seed}/{dimension}d_{signal_name}_{bandlimit}_{seed}.png')
                        plt.close()
                    np.save(f'{target_dir}{signal_name}/{seed}/{dimension}d_{signal_name}_{bandlimit}_{seed}.npy', signal.signal)
                    with open(target_dir+f'{signal_name}/{seed}/estimated_time_{seed}.csv', 'a', newline='') as f:
                        writer_iter = csv.DictWriter(f, fieldnames=fieldnames)
                        writer_iter.writerow({
                            'seed': seed, 
                            'bandlimits': bandlimit,
                            'time': time_spent, 
                        })

                


