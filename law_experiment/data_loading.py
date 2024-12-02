import numpy as np
import pickle
import torch
import torch.utils.data as data

class lawDataset(data.Dataset):
    def __init__(self, law_data, type="train"):
        """
        0 - 2: S
        2 - 7: R
        7 - 1007: K
        1007: G
        1008: L
        1009: F
        """
        self.bG = law_data["bG"]
        self.wGK = law_data["wGK"]
        self.wGR = law_data["wGR"]
        self.wGS = law_data["wGS"]
        self.sigma_1 = law_data["sigma_1"]
        self.sigma_2 = law_data["sigma_2"]
        
        self.bL = law_data["bL"]
        self.wLK = law_data["wLK"]
        self.wLR = law_data["wLR"]
        self.wLS = law_data["wLS"]
        
        self.wFK = law_data["wFK"]
        self.wFR = law_data["wFR"]
        self.wFS = law_data["wFS"]
        
        if type == "train":
            self.S = law_data["train_S"][:int(0.75 * len(law_data["train_S"]))]
            self.R = law_data["train_R"][:int(0.75 * len(law_data["train_S"]))]
            self.K = law_data["train_K"][:int(0.75 * len(law_data["train_S"]))]
            self.G = law_data["train_G"][:int(0.75 * len(law_data["train_S"]))][:, np.newaxis]
            self.L = law_data["train_L"][:int(0.75 * len(law_data["train_S"]))][:, np.newaxis]
            self.F = law_data["train_F"][:int(0.75 * len(law_data["train_S"]))][:, np.newaxis]
            
        elif type == "valid":
            self.S = law_data["train_S"][int(0.75 * len(law_data["train_S"])):]
            self.R = law_data["train_R"][int(0.75 * len(law_data["train_S"])):]
            self.K = law_data["train_K"][int(0.75 * len(law_data["train_S"])):]
            self.G = law_data["train_G"][int(0.75 * len(law_data["train_S"])):][:, np.newaxis]
            self.L = law_data["train_L"][int(0.75 * len(law_data["train_S"])):][:, np.newaxis]
            self.F = law_data["train_F"][int(0.75 * len(law_data["train_S"])):][:, np.newaxis]
            
        else:
            self.S = law_data["test_S"][:]
            self.R = law_data["test_R"]
            self.K = law_data["test_K"]
            self.G = law_data["test_G"][:, np.newaxis]
            self.L = law_data["test_L"][:, np.newaxis]
            self.F = law_data["test_F"][:, np.newaxis]
            
        self.data = torch.cat([torch.Tensor(self.S), torch.Tensor(self.R), torch.Tensor(self.K), torch.Tensor(self.G),
                               torch.Tensor(self.L), torch.Tensor(self.F)], dim=-1)
    
    def __getitem__(self, index):
        return self.data[index]
    
    def __len__(self):
        return len(self.data)
