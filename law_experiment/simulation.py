import argparse
import matplotlib
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pickle
import random
import torch
import torch.nn as nn
import torch.optim as optim

#from data_generate import data_generation
from data_loading import lawDataset
from pathlib import Path
from scipy.stats import gaussian_kde
from torch.utils.data import DataLoader

#from pylab import *

def GPA(S, R, K, dataset):
    return torch.mm(S, torch.Tensor(dataset.wGS).unsqueeze(1).cuda()) + torch.mm(R, torch.Tensor(dataset.wGR).unsqueeze(1).cuda()) + torch.tensor(dataset.wGK).cuda() * K + dataset.bG

def SAT(S, R, K, dataset):
    return torch.exp(torch.mm(S, torch.Tensor(dataset.wLS).unsqueeze(1).cuda()) + torch.mm(R, torch.Tensor(dataset.wLR).unsqueeze(1).cuda()) + torch.tensor(dataset.wLK).cuda() * K + dataset.bL)

def FYA(S, R, K, dataset):
    return torch.mm(S, torch.Tensor(dataset.wFS).unsqueeze(1).cuda()) + torch.mm(R, torch.Tensor(dataset.wFR).unsqueeze(1).cuda()) + torch.tensor(dataset.wFK).cuda() * K

class UFdecision(nn.Module):
    def __init__(self, input_dim):
        super(UFdecision, self).__init__()
        #self.layer = nn.Linear(input_dim, 1)
        self.layer = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        out = self.layer(x)
        return out

def UFtrain(loader, vloader, model, loss_fn, optimizer, total_epochs, eta):
    losses = []
    valid_losses = []
    
    for __ in range(total_epochs):
        model.train()
        running_loss = 0
        for __, data in enumerate(loader):
            optimizer.zero_grad()
            R = data[:, 2 : 7].detach().clone().cuda()
            G = data[:, 1007].unsqueeze(1).detach().clone().cuda()
            L = data[:, 1008].unsqueeze(1).detach().clone().cuda()
            F = data[:, 1009].unsqueeze(1).detach().clone().cuda()
            
            out = model(torch.cat([R, G, L], dim=1))
            loss = loss_fn(out, F)
            loss.backward()
            running_loss += loss.item()
            optimizer.step()
        losses.append(running_loss / len(loader))
    
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig("images/UF_train_loss.png")
    plt.close()

def UFtest(loader, model, loss_fn, eta):
    model.eval()
    
    original_unfairs = []
    unfairs = []
    running_loss = 0
    
    Y_current = []
    cY_current = []
    
    Y_prime = []
    cY_prime = []
    
    for i, data in enumerate(loader):
        S = data[:, 0 : 2].detach().clone().cuda()
        R = data[:, 2 : 7].detach().clone().cuda()
        G = data[:, 1007].unsqueeze(1).detach().clone().cuda()
        L = data[:, 1008].unsqueeze(1).detach().clone().cuda()
        target = data[:, 1009].unsqueeze(1).detach().clone().cuda()
        
        with torch.no_grad():
            out = model(torch.cat([R, G, L], dim=1))
            loss = loss_fn(out, target)
            running_loss += loss.item()
        
        o_fair = torch.zeros(size=(S.size()[0], 1)).cuda()
        fair = torch.zeros(size=(S.size()[0], 1)).cuda()
        for k_index in range(500):
            K = data[:, 10 + k_index].unsqueeze(1).detach().clone().cuda()
            cK = K.detach().clone().cuda()
            
            #F_check = torch.normal(FYA(1 - S, R, K, loader.dataset), 1)
            F_check = FYA(1 - S, R, K, loader.dataset)
            #F = torch.normal(FYA(S, R, K, loader.dataset), 1)
            F = FYA(S, R, K, loader.dataset)
            o_fair += torch.abs(F - F_check).detach().clone()
            
            if i == 0:
                Y_current.append(F[0].item())
                cY_current.append(F_check[0].item())
            
            #G = torch.normal(GPA(S, R, K, loader.dataset), loader.dataset.sigma)
            G = GPA(S, R, K, loader.dataset)
            #L = torch.poisson(SAT(S, R, K, loader.dataset))
            L = SAT(S, R, K, loader.dataset)
            G.requires_grad = True
            L.requires_grad = True
            out = model(torch.cat([R, G, L], dim=1))
            gradient_dummy = torch.ones_like(out)
            out.backward(gradient=gradient_dummy)
            grad_K = loader.dataset.wGK * G.grad + loader.dataset.wLK * L.grad
            f_K = K + eta * grad_K
            #f_F = torch.normal(FYA(S, R, f_K, loader.dataset), 1)
            f_F = FYA(S, R, f_K, loader.dataset)
            
            #cG = torch.normal(GPA(1 - S, R, cK, loader.dataset), loader.dataset.sigma)
            cG = GPA(1 - S, R, cK, loader.dataset)
            #cL = torch.poisson(SAT(1 - S, R, cK, loader.dataset))
            cL = SAT(1 - S, R, cK, loader.dataset)
            cG.requires_grad = True
            cL.requires_grad = True
            cout = model(torch.cat([R, cG, cL], dim=1))
            cgradient_dummy = torch.ones_like(cout)
            cout.backward(gradient=cgradient_dummy)
            grad_cK = loader.dataset.wGK * cG.grad + loader.dataset.wLK * L.grad
            f_cK = cK + eta * grad_cK
            #f_cF = torch.normal(FYA(1 - S, R, f_cK, loader.dataset), 1)
            f_cF = FYA(1 - S, R, f_cK, loader.dataset)
            
            fair += torch.abs(f_F - f_cF).detach().clone()
            
            if i == 0:
                Y_prime.append(f_F[1].item())
                cY_prime.append(f_cF[1].item())
        
        o_fair = o_fair / 500
        fair = fair / 500
        
        original_unfairs.extend([o_fair[i].item() for i in range(o_fair.size()[0])])
        unfairs.extend([fair[i].item() for i in range(fair.size()[0])])
        
    original_unfairness = np.mean(original_unfairs)
    unfairness = np.mean(unfairs)
    loss = running_loss / len(loader)
    
    with open("UF_Y.pkl", "wb") as f:
        pickle.dump(Y_prime, f)
    with open("UF_cY.pkl", "wb") as f:
        pickle.dump(cY_prime, f)
    with open("current_Y.pkl", "wb") as f:
        pickle.dump(Y_current, f)
    with open("current_cY.pkl", "wb") as f:
        pickle.dump(cY_current, f)
    
    return original_unfairness, unfairness, loss

def UFexperiment(loaders, total_epochs, eta):
    train_loader, valid_loader, test_loader = loaders[0], loaders[1], loaders[2]
    model = UFdecision(input_dim=7).cuda()
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.99))
    UFtrain(train_loader, valid_loader, model, loss_fn, optimizer, total_epochs, eta)
    original_unfairness, unfairness, test_loss = UFtest(test_loader, model, loss_fn, eta)
    print("original_unfairnss = {}, unfairness = {}, test_loss = {}".format(original_unfairness, unfairness, test_loss))
    return original_unfairness, unfairness, test_loss

class CFdecision(nn.Module):
    def __init__(self, input_dim):
        super(CFdecision, self).__init__()
        #self.layer = nn.Linear(input_dim, 1)
        self.layer = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        out = self.layer(x)
        return out

def CFtrain(loader, vloader, model, loss_fn, optimizer, total_epochs, eta):
    losses = []
    
    for __ in range(total_epochs):
        model.train()
        running_loss = 0
        for __, data in enumerate(loader):
            optimizer.zero_grad()
            K = data[:, 10 + random.randint(0, 500)].unsqueeze(1).detach().clone().cuda()
            S = data[:, 0 : 2].detach().clone().cuda()
            R = data[:, 2 : 7].detach().clone().cuda()
            target = data[:, 1009].unsqueeze(1).detach().clone().cuda()
            
            out = model(torch.cat([R, K], dim=1))
            loss = loss_fn(out, target)
            loss.backward()
            running_loss += loss.item()
            optimizer.step()
        losses.append(running_loss / len(loader))
    
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig("images/CF_train_loss.png")
    plt.close()

def CFtest(loader, model, loss_fn, eta):
    model.eval()
    
    original_unfairs = []
    unfairs = []
    running_loss = 0
    
    Y_prime = []
    cY_prime = []
    
    for i, data in enumerate(loader):
        S = data[:, 0 : 2].detach().clone().cuda()
        R = data[:, 2 : 7].detach().clone().cuda()
        target = data[:, 1009].unsqueeze(1).detach().clone().cuda()
        
        o_fair = torch.zeros(size=(S.size()[0], 1)).cuda()
        fair = torch.zeros(size=(S.size()[0], 1)).cuda()
        for k_index in range(500):
            K = data[:, k_index].unsqueeze(1).detach().clone().cuda()
            cK = K.detach().clone().cuda()
            
            with torch.no_grad():
                out = model(torch.cat([R, K], dim=1))
                loss = loss_fn(out, target)
                running_loss += loss.item()
                
                #F = torch.normal(FYA(S, R, K, loader.dataset), 1)
                F = FYA(S, R, K, loader.dataset)
                #F_check = torch.normal(FYA(1 - S, R, K, loader.dataset), 1)
                F_check = FYA(1 - S, R, cK, loader.dataset)
                
                o_fair += torch.abs(F - F_check).detach().clone()
        
            K.requires_grad = True
            out = model(torch.cat([R, K], dim=1))
            gradient_dummy = torch.ones_like(out)
            out.backward(gradient_dummy)
            f_K = K + eta * K.grad
            #f_F = torch.normal(FYA(S, R, f_K, loader.dataset), 1)
            f_F = FYA(S, R, f_K, loader.dataset)
            
            cK.requires_grad = True
            cout = model(torch.cat([R, cK], dim=1))
            cgradient_dummy = torch.ones_like(cout)
            cout.backward(cgradient_dummy)
            f_cK = cK + eta * cK.grad
            #f_cF = torch.normal(FYA(1 - S, R, f_cK, loader.dataset), 1)
            f_cF = FYA(1 - S, R, f_cK, loader.dataset)
            
            fair += torch.abs(f_F - f_cF).detach().clone()
            
            if i == 0:
                Y_prime.append(f_F[1].item())
                cY_prime.append(f_cF[1].item())
        
        o_fair = o_fair / 500
        fair = fair / 500
        
        original_unfairs.extend([o_fair[i].item() for i in range(o_fair.size()[0])])
        unfairs.extend([fair[i].item() for i in range(fair.size()[0])])
    
    original_unfairness = np.mean(original_unfairs)
    unfairness = np.mean(unfairs)
    loss = running_loss / (500 * len(loader))
    
    with open("CF_Y.pkl", "wb") as f:
        pickle.dump(Y_prime, f)
    with open("CF_cY.pkl", "wb") as f:
        pickle.dump(cY_prime, f)
    
    return original_unfairness, unfairness, loss

def CFexperiment(loaders, total_epochs, eta):
    train_loader, valid_loader, test_loader = loaders[0], loaders[1], loaders[2]
    model = CFdecision(input_dim=6).cuda()
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.99))
    CFtrain(train_loader, valid_loader, model, loss_fn, optimizer, total_epochs, eta)
    original_unfairness, unfairness, test_loss = CFtest(test_loader, model, loss_fn, eta)
    print("original_unfairnss = {}, unfairness = {}, test_loss = {}".format(original_unfairness, unfairness, test_loss))
    return original_unfairness, unfairness, test_loss

class DFdecision(nn.Module):
    def __init__(self, input_dim):
        super(DFdecision, self).__init__()
        self.w1 = torch.Tensor([0]).cuda()
        self.w2 = nn.Parameter(torch.randn(1))
        self.w3 = nn.Parameter(torch.randn(1))
        
        #self.layer = nn.Linear(input_dim, 1)
        self.layer = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, cy, u):
        h1 = self.w1 * cy ** 2 + self.w2 * cy + self.w3
        h2 = self.layer(u)
        out = h1 + h2
        return out

def DFtrain(loader, vloader, model, loss_fn, optimizer, total_epoch, eta):
    losses = []
    
    for __ in range(total_epoch):
        model.train()
        running_loss = 0
        for __, data in enumerate(loader):
            optimizer.zero_grad()
            K = data[:, 10 + random.randint(0, 500)].unsqueeze(1).detach().clone().cuda()
            #K = torch.mean(data[:, 10 + K_index], dim=1).unsqueeze(1).detach().clone().cuda()
            S = data[:, 0 : 2].detach().clone().cuda()
            R = data[:, 2 : 7].detach().clone().cuda()
            F = data[:, 1009].unsqueeze(1).detach().cuda()
            
            #F_check = torch.normal(FYA(1 - S, R, K, loader.dataset), 1)
            F_check = FYA(1 - S, R, K, loader.dataset)
            #F = torch.normal(FYA(S, R, K, loader.dataset), 1)
            #F = FYA(S, R, K, loader.dataset)
            
            out = model(F_check, torch.cat([R, K], dim=1))
            loss = loss_fn(out, F)
            loss.backward()
            running_loss += loss.item()
            optimizer.step()
        losses.append(running_loss / len(loader))
    
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig("images/DF_train_loss.png")
    plt.close()

def DFtest(loader, model, loss_fn, eta, ratio):
    model.eval()
    
    original_unfairs = []
    unfairs = []
    running_loss = 0
    
    Y_prime = []
    cY_prime = []
    
    for i, data in enumerate(loader):
        S = data[:, 0 : 2].detach().clone().cuda()
        R = data[:, 2 : 7].detach().clone().cuda()
        target = data[:, 1009].unsqueeze(1).detach().clone().cuda()
        
        o_fair = torch.zeros(size=(S.size()[0], 1)).cuda()
        fair = torch.zeros(size=(S.size()[0], 1)).cuda()
        for K_index in range(500):
            K = data[:, 10 + K_index].unsqueeze(1).detach().clone().cuda()
            #K = torch.mean(data[:, 10 : 510], dim=1).unsqueeze(1).detach().clone().cuda()
            cK = K.detach().clone().cuda()
            
            with torch.no_grad():
                #F_check = torch.normal(FYA(1 - S, R, K, loader.dataset), 1)
                F_check = FYA(1 - S, R, K, loader.dataset)
                out = model(F_check, torch.cat([R, K], dim=1))
                loss = loss_fn(out, target)
                running_loss += loss.item()
                
                #F = torch.normal(FYA(S, R, K, loader.dataset), 1)
                F = FYA(S, R, K, loader.dataset)
                
                o_fair += torch.abs(F - F_check).detach().clone()
            
            K.requires_grad = True
            eplision = torch.randn(1).cuda()
            #F_check = FYA(1 - S, R, K, loader.dataset) + eplision
            F_check = FYA(1 - S, R, K, loader.dataset)
            out = model(F_check, torch.cat([R, K], dim=1))
            gradient_dummy = torch.ones_like(out)
            out.backward(gradient=gradient_dummy)
            f_K = K + eta * K.grad
            #f_F = torch.normal(FYA(S, R, f_K, loader.dataset), 1)
            f_F = FYA(S, R, f_K, loader.dataset)
            
            cK.requires_grad = True
            ceplision = torch.randn(1).cuda()
            #cF_check = FYA(S, R, cK, loader.dataset) + ceplision
            cF_check = FYA(S, R, cK, loader.dataset)
            cout = model(cF_check, torch.cat([R, cK], dim=1))
            cgradient_dummy = torch.ones_like(cout)
            cout.backward(gradient=cgradient_dummy)
            f_cK = cK + eta * cK.grad
            #f_cF = torch.normal(FYA(1 - S, R, f_cK, loader.dataset), 1)
            f_cF = FYA(1 - S, R, f_cK, loader.dataset)
            
            fair += torch.abs(f_F - f_cF).detach().clone()
            
            if i == 0:
                Y_prime.append(f_F[1].item())
                cY_prime.append(f_cF[1].item())
        
        o_fair = o_fair / 500
        fair = fair / 500
        
        original_unfairs.extend([o_fair[i].item() for i in range(o_fair.size()[0])])
        unfairs.extend([fair[i].item() for i in range(fair.size()[0])])
    
    original_unfairness = np.mean(original_unfairs)
    unfairness = np.mean(unfairs)
    loss = running_loss / (500 * len(loader))
    
    with open("DF_Y_{}.pkl".format(ratio), "wb") as f:
        pickle.dump(Y_prime, f)
    with open("DF_cY_{}.pkl".format(ratio), "wb") as f:
        pickle.dump(cY_prime, f)
    
    return original_unfairness, unfairness, loss

def DFexperiment(loaders, total_epochs, eta, ratio):
    train_loader, valid_loader, test_loader = loaders[0], loaders[1], loaders[2]
    model = DFdecision(input_dim=6)
    model.cuda()
    model.w1 = torch.tensor(1 / (2 * ratio * eta * train_loader.dataset.wFK ** 2))
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.99))
    DFtrain(train_loader, valid_loader, model, loss_fn, optimizer, total_epochs, eta)
    original_unfairness, unfairness, test_loss = DFtest(test_loader, model, loss_fn, eta, ratio)
    print("original_unfairness = {}, unfairness = {}, test_loss = {}".format(original_unfairness, unfairness, test_loss))
    return original_unfairness, unfairness, test_loss

#def compute_table():
#    seeds = [42, 43, 44, 45, 46]
#    for seed in seeds:
#        with open("./datas/data_{}.pkl".format(seed), "rb") as f:
#            law_data = pickle.load(f)
        
#        train_set = lawDataset(law_data, type="train")
#        valid_set = lawDataset(law_data, type="valid")
#        test_set = lawDataset(law_data, type="test")
#        train_loader = DataLoader(train_set, batch_size=128, shuffle=True)
#        valid_loader = DataLoader(valid_set, batch_size=128, shuffle=False)
#        test_loader = DataLoader(test_set, batch_size=128, shuffle=False)
#        loaders = [train_loader, valid_loader, test_loader]
        
#        total_epochs = 100
#        eta = 10
        
#        for i in range(0, 3):
#            if i == 0:
#                UFexperiment(loaders, total_epochs, eta)
#            elif i == 1:
#                CFexperiment(loaders, total_epochs, eta)
#            elif i == 2:
#                DFexperiment(loaders, total_epochs, eta, ratio=1)
#        print("\n")
        
def compute_table():
    o_fairnesses = np.zeros((5, 4))
    fairnesses = np.zeros((5, 4))
    losses = np.zeros((5, 4))
    seeds = [42, 43, 44, 45, 46]
    for seed in seeds:
        with open("./datas/data_{}.pkl".format(seed), "rb") as f:
            law_data = pickle.load(f)
        train_set = lawDataset(law_data, type="train")
        valid_set = lawDataset(law_data, type="valid")
        test_set = lawDataset(law_data, type="test")
        train_loader = DataLoader(train_set, batch_size=128, shuffle=True)
        valid_loader = DataLoader(valid_set, batch_size=128, shuffle=False)
        test_loader = DataLoader(test_set, batch_size=128, shuffle=False)
        loaders = [train_loader, valid_loader, test_loader]
            
        total_epochs = 100
        eta = 10
        for i in range(4):
            if i == 0:
                original_unfairness, fairness, loss = UFexperiment(loaders, total_epochs, eta)
            elif i == 1:
                original_unfairness, fairness, loss = CFexperiment(loaders, total_epochs, eta)
            elif i == 2:
                original_unfairness, fairness, loss = DFexperiment(loaders, total_epochs, eta, 1)
            elif i == 3:
                original_unfairness, fairness, loss = DFexperiment(loaders, total_epochs, eta, 2)
            o_fairnesses[seed - 42, i], fairnesses[seed - 42, i], losses[seed - 42, i] = original_unfairness, fairness, loss
    print("UF original_unfairess = {:.3f} + {:.3f}, unfairness = {:.3f} + {:.3f}, loss = {:.3f} + {:.3f}".format(
        np.mean(o_fairnesses[:, 0]), np.std(o_fairnesses[:, 0]), np.mean(fairnesses[:, 0]), np.std(fairnesses[:, 0]),
        np.mean(losses[:, 0]), np.std(losses[:, 0])
    ))
    print("CF original_unfairess = {:.3f} + {:.3f}, unfairness = {:.3f} + {:.3f}, loss = {:.3f} + {:.3f}".format(
        np.mean(o_fairnesses[:, 1]), np.std(o_fairnesses[:, 1]), np.mean(fairnesses[:, 1]), np.std(fairnesses[:, 1]),
        np.mean(losses[:, 1]), np.std(losses[:, 1])
    ))
    print("DF original_unfairess = {:.3f} + {:.3f}, unfairness = {:.3f} + {:.3f}, loss = {:.3f} + {:.3f}".format(
        np.mean(o_fairnesses[:, 2]), np.std(o_fairnesses[:, 2]), np.mean(fairnesses[:, 2]), np.std(fairnesses[:, 2]),
        np.mean(losses[:, 2]), np.std(losses[:, 2])
    ))
    print("DF original_unfairess = {:.3f} + {:.3f}, unfairness = {:.3f} + {:.3f}, loss = {:.3f} + {:.3f}".format(
        np.mean(o_fairnesses[:, 3]), np.std(o_fairnesses[:, 3]), np.mean(fairnesses[:, 3]), np.std(fairnesses[:, 3]),
        np.mean(losses[:, 3]), np.std(losses[:, 3])
    ))

def drawDensity():
    with open("./datas/data_44.pkl", "rb") as f:
        law_data = pickle.load(f)
    
    train_set = lawDataset(law_data, type="train")
    valid_set = lawDataset(law_data, type="valid")
    test_set = lawDataset(law_data, type="test")
    train_loader = DataLoader(train_set, batch_size=128, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=128, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=128, shuffle=False)
    loaders = [train_loader, valid_loader, test_loader]
    
    total_epochs = 100
    eta = 10
    
    for i in range(4):
        if i == 0:
            UFexperiment(loaders, total_epochs, eta)
        elif i == 1:
            CFexperiment(loaders, total_epochs, eta)
        elif i == 2:
            DFexperiment(loaders, total_epochs, eta, 1)
        else:
            DFexperiment(loaders, total_epochs, eta, 2)
                
    with open("UF_Y.pkl", "rb") as f:
        UF_Y = pickle.load(f)
    with open("UF_cY.pkl", "rb") as f:
        UF_cY = pickle.load(f)
    with open("CF_Y.pkl", "rb") as f:
        CF_Y = pickle.load(f)
    with open("CF_cY.pkl", "rb") as f:
        CF_cY = pickle.load(f)
    with open("DF_Y_1.pkl", "rb") as f:
        DF_Y = pickle.load(f)
    with open("DF_cY_1.pkl", "rb") as f:
        DF_cY = pickle.load(f)
    with open("current_Y.pkl", "rb") as f:
        current_Y = pickle.load(f)
    with open("current_cY.pkl", "rb") as f:
        current_cY = pickle.load(f)
    with open("DF_Y_2.pkl", "rb") as f:
        DF_Y_2 = pickle.load(f)
    with open("DF_cY_2.pkl", "rb") as f:
        DF_cY_2 = pickle.load(f)
    
    factuals = [current_Y, UF_Y, CF_Y, DF_Y_2, DF_Y]
    counters = [current_cY, UF_cY, CF_cY, DF_cY_2, DF_cY]
    
    print("current {} ~ {}".format(min(current_Y), max(current_cY)))
    print("UF {} ~ {}".format(min(UF_Y), max(UF_Y)))
    print("CF {} ~ {}".format(min(CF_Y), max(CF_Y)))
    print("DF_2 {} ~ {}".format(min(DF_Y_2), max(DF_cY_2)))
    print("DF {} ~ {}".format(min(DF_Y), max(DF_Y)))

    plt.rcParams['font.size'] = '45'
    plt.rcParams["font.family"] = "normal"
    #plt.rcParams['text.usetex'] = True

    def plot_single_graph(i, factual_data, counter_data, ax, title):
        kde1 = gaussian_kde(factual_data)
        kde2 = gaussian_kde(counter_data)
        
        if i == 0:
            x = np.linspace(min(current_Y), max(current_cY), 100)
        elif i == 1:
            x = np.linspace(min(UF_Y), max(UF_Y), 100)
        elif i == 2:
            x = np.linspace(min(CF_Y), max(CF_Y), 100)
        elif i == 3:
            x = np.linspace(min(DF_Y_2), max(DF_Y_2), 100)
        elif i == 4:
            x = np.linspace(min(DF_Y), max(DF_Y), 100)
        density1 = kde1(x)
        density2 = kde2(x)

        ax.plot(x, density1, color="b", label="factual")
        ax.plot(x, density2, color="r", label="counterfactual")

        ax.fill_between(x, density1, color='b', alpha=0.2)
        ax.fill_between(x, density2, color='r', alpha=0.2)
        
        #ax.legend()
        if i != 0:
            #ax.set_xlabel("$Y'$")
            ax.set_xlabel("Y'")
        else:
            #ax.set_xlabel("$Y$")
            ax.set_xlabel("Y")
        ax.set_ylabel("density")
        ax.set_title(title)

    fig, axs = plt.subplots(1, 5, figsize=(50, 10.5))


    #titles = ["Baseline", "CA", "ICA", "CE", "CR", "Ours"]
    #titles = ["Current", "UF", "CF", r"DF($p_{1}=\frac{T}{4}$)", r"DF($p_{1}=\frac{T}{2}$)"]
    titles = ["Current", "UF", "CF", "LCF (50 %)", "LCF (100 %)"]
    for i, title in enumerate(titles):
        plot_single_graph(i, factuals[i], counters[i], axs[i], title)
    #plt.subplots_adjust(left=None, bottom=0.15, right=None, top=0.85, wspace=0.3, hspace=0.8)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.3, hspace=0.5)
    lineA = mlines.Line2D([], [], color='b', label='factual')
    lineB = mlines.Line2D([], [], color='r', label='counter\nfactual')
    # Create a legend for the whole figure
    fig.legend(handles=[lineA, lineB], loc='center right', ncol=1, framealpha=1)

    plt.show()
    plt.savefig("density.pdf")
    #plt.savefig("density_cvae.pdf")

def plot_trade_off():
    unfairs = np.zeros((3, 6))
    losses = np.zeros((3, 6))

    with open("./datas/data_42.pkl", "rb") as f:
        law_data = pickle.load(f)
    
    train_set = lawDataset(law_data, type="train")
    valid_set = lawDataset(law_data, type="valid")
    test_set = lawDataset(law_data, type="test")
    train_loader = DataLoader(train_set, batch_size=128, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=128, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=128, shuffle=False)
    loaders = [train_loader, valid_loader, test_loader]
    
    
    etas = [15, 10, 5]
    ratios = [1, 2, 4, 16, 32, 256]
    
    for i, eta in enumerate(etas):
        for j, ratio in enumerate(ratios):
            total_epochs = 50
            __, unfairness, loss = DFexperiment(loaders, total_epochs, eta, ratio)
            unfairs[i, j] = unfairness
            losses[i, j] = loss
    
    font = {"family": "normal", "size": 13}
    matplotlib.rc("font", **font)

    fig = plt.figure(figsize=(5, 5))
    p1, = plt.plot(losses[0], unfairs[0], "r*-", label="$\eta=15$", linewidth=2)
    p2, = plt.plot(losses[1], unfairs[1], "c>:", label="$\eta=10$", linewidth=2)
    p3, = plt.plot(losses[2], unfairs[2], "y^-", label="$\eta=5$", linewidth=2)
    #p4, = plt.plot(losses[3], unfairs[3], "bx--", label="$\eta=0.5$", linewidth=2)
    plt.legend()

    plt.xlabel("MSE")
    plt.ylabel("AFCE")
    plt.grid()
    plt.tight_layout()
    plt.show()
    fig.savefig("trade_off_curves.png")

def main():
    seeds = [42, 43, 44, 45, 46]
    for seed in seeds:
        check_data = Path("./datas/data_{}.pkl".format(seed))
        if check_data.is_file():
            pass
        else:
            #data_generation(seed)
            pass
    
    #compute_table()
    drawDensity()
    #plot_trade_off()

if __name__ == "__main__":
    main()