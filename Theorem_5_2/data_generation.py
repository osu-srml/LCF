import numpy as np
import torch
import torch.utils.data as data

from sklearn.model_selection import train_test_split

class simData(object):
    def __init__(self, N, d):
        """
        generate simulated data with a linear causal model
        """
        np.random.seed(42)
        self.wux = np.random.uniform(0, 1, size=(d, 1))
        self.wax = np.random.uniform(0, 1, size=(d, 1))
        self.wxy = np.random.uniform(0, 1, size=(d, 1))
        self.wuy = np.random.uniform(0, 1)
        self.N = N
        self.d = d
        print("wux = {}".format(self.wux))
        print("wax = {}".format(self.wax))
        print("wxy = {}".format(self.wxy))
        print("wuy = {}".format(self.wuy))
        
        self.generateData()
    
    def generateData(self):
        """
        0: a
        1 - d + 1: ux
        d + 1 - 2 * d + 1: x
        2 * d + 1: uy
        2 * d + 2: y
        """
        self.a = np.random.binomial(n=1, p=0.5, size=(self.N, 1))
        self.ux = np.random.uniform(0, 1, size=(self.N, self.d))
        self.x = self.ux * self.wux.T + self.a * self.wax.T
        self.uy = np.random.uniform(0, 1, size=(self.N, 1))
        self.y = np.dot(self.x, self.wxy) + self.uy * self.wuy
        self.data = np.concatenate([self.a, self.ux, self.x, self.uy, self.y], axis=1)
    
    def splitData(self, seed):
        self.train_data, val_data = train_test_split(self.data, test_size=0.8, random_state=seed)
        self.validate_data, self.test_data = train_test_split(val_data, test_size=0.5, random_state=seed)
        return self.train_data, self.validate_data, self.test_data

class simDataset(data.Dataset):
    def __init__(self, simdata, type="train"):
        if type == "train":
            self.data = torch.tensor(simdata.train_data, dtype=torch.float32)
        elif type == "valid":
            self.data = torch.tensor(simdata.validate_data, dtype=torch.float32)
        else:
            self.data = torch.tensor(simdata.test_data, dtype=torch.float32)
        
    def __getitem__(self, index):
        return self.data[index]
    
    def __len__(self):
        return len(self.data)