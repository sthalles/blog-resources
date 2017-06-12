from urllib.request import urlretrieve
from os.path import isfile, isdir
from tqdm import tqdm
import os
from scipy.io import loadmat
import numpy as np
import matplotlib.cm as cm
import subprocess
import glob

class Dataset:
    
    def __init__(self, dataset_name="svhn", data_dir='data/', shuffle=False):
        self.dataset_name = dataset_name
        if not isdir(data_dir):
            os.makedirs(data_dir)
        self.data_dir = data_dir
        self.dataset = self._load_dataset()
        self.length = self.dataset.shape[0]
        self.shuffle = shuffle
        
    def _load_dataset(self):
        """
        Download and return the dataset
        """
        if self.dataset_name == 'svhn':
            self._download_svhn_if_necessary()
            trainset = loadmat(self.data_dir + 'train_32x32.mat')
            trainset = np.rollaxis(trainset['X'], 3)

            testset = loadmat(self.data_dir + 'test_32x32.mat')
            testset = np.rollaxis(testset['X'], 3)
            return np.concatenate((trainset, testset), axis=0)
       
        elif self.dataset_name == 'mnist':
            from tensorflow.examples.tutorials.mnist import input_data
            mnist = input_data.read_data_sets(self.data_dir + 'MNIST_data')
            train_set, test_set = mnist.train.images, mnist.test.images
            trainset = np.concatenate((train_set, test_set), axis=0)
            trainset = trainset.reshape([-1,28,28,1])
            return np.pad(trainset, ((0,0),(2,2),(2,2),(0,0)), 'constant', constant_values=0.)
            
    
    def images(self):
        """
        Returns the images
        """
        return self.dataset
    
    def _download_svhn_if_necessary(self):
        """
        Downloads the dataset if necessary
        """
        if not isfile(self.data_dir + "train_32x32.mat"):
            with DLProgress(unit='B', unit_scale=True, miniters=1, desc='SVHN Training Set') as pbar:
                urlretrieve(
                    'http://ufldl.stanford.edu/housenumbers/train_32x32.mat',
                    self.data_dir + 'train_32x32.mat',
                    pbar.hook)

        if not isfile(self.data_dir + "test_32x32.mat"):
            with DLProgress(unit='B', unit_scale=True, miniters=1, desc='SVHN Training Set') as pbar:
                urlretrieve(
                    'http://ufldl.stanford.edu/housenumbers/test_32x32.mat',
                    self.data_dir + 'test_32x32.mat',
                    pbar.hook)
     
    def _scale(self, x, feature_range=(-1, 1)):
        """
        Scale the images to have pixel values between -1 and 1
        """
        if self.dataset_name == 'svhn':
            # scale to (0, 1), for mnist it is not necessary
            x = ((x - x.min())/(255. - x.min()))

        # scale to feature_range
        min, max = feature_range
        x = x * (max - min) + min
        return x
    
    def next_batch(self, batch_size):
        """
        Return the next batch for the training loop
        """
        if self.shuffle:
            print("Dataset shuffled successfully.") 
            idx = np.arange(self.length)
            np.random.shuffle(idx)
            self.dataset = self.dataset[idx]
        
        for ii in range(0, self.length, batch_size):
            x = self.dataset[ii:ii+batch_size]
            yield self._scale(x)
            

def generate_video(video_name, folder="./video"):
    cwd = os.getcwd()
    if not isdir(folder):
        os.makedirs(folder)
    os.chdir(folder)
    subprocess.call([
        'ffmpeg', '-framerate', '8', '-i', 'file%02d.png', '-r', '30', '-pix_fmt', 'yuv420p',
        'video_' + video_name + '.mp4'
    ])
    for file_name in glob.glob("*.png"):
        os.remove(file_name)
    os.chdir(cwd)

class DLProgress(tqdm):
    last_block = 0

    def hook(self, block_num=1, block_size=1, total_size=None):
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num
