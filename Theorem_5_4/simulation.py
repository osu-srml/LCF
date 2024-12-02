import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.optim as optim

from data_generation import simData, simDataset
from scipy.stats import gaussian_kde
from torch.utils.data import DataLoader

from pylab import *

def get_x(ux, a, wux, wax):
    x = (ux * torch.tensor(wux.T, dtype=torch.float32).cuda() + torch.tensor(wax.T, dtype=torch.float32).cuda()) * a
    return x

def get_y(uy, x, wxy, wuy):
    y = torch.mm(x, torch.tensor(wxy, dtype=torch.float32).cuda()) + uy * torch.tensor(wuy, dtype=torch.float32).cuda()
    return y

class UFdecision(nn.Module):
    def __init__(self, input_dim):
        super(UFdecision, self).__init__()
        self.layer = nn.Linear(input_dim, 1)
    
    def forward(self, x):
        out = self.layer(x)
        return out

def UFtrain(loader, vloader, model, loss_fn, optimizer, total_epochs, eta, simdata):
    losses = []
    valid_losses = []
    
    d = simdata.d
    
    for __ in range(total_epochs):
        model.train()
        running_loss = 0
        for __, data in enumerate(loader):
            optimizer.zero_grad()
            x = data[:, d + 1 : 2 * d + 1].detach().clone().cuda()
            y = data[:, 2 * d + 2].unsqueeze(1).detach().clone().cuda()
            
            out = model(x)
            loss = loss_fn(out, y)
            loss.backward()
            running_loss += loss.item()
            optimizer.step()
        losses.append(running_loss / len(loader))
        
        model.eval()
        running_loss = 0
        for __, data in enumerate(vloader):
            with torch.no_grad():
                x = data[:, d + 1 : 2 * d + 1].detach().clone().cuda()
                y = data[:, 2 * d + 2].unsqueeze(1).detach().clone().cuda()
                
                out = model(x)
                loss = loss_fn(out, y)
                running_loss += loss.item()
        valid_losses.append(running_loss / len(vloader))
    
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig("images/UF_train_loss.png")
    plt.close()
    plt.plot(valid_losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig("images/UF_valid_loss.png")
    plt.close()

def UFtest(loader, model, loss_fn, eta, simdata):
    model.eval()
    
    d = simdata.d
    
    original_unfairs = []
    unfairs = []
    running_loss = 0
    
    Y_current = []
    cY_current = []
    
    Y_prime = []
    cY_prime = []
    
    for i, data in enumerate(loader):
        a = data[:, 0].unsqueeze(1).detach().clone().cuda()
        target = data[:, 2 * d + 2].unsqueeze(1).detach().clone().cuda()
        x = data[:, d + 1 : 2 * d + 1].detach().clone().cuda()
        
        with torch.no_grad():
            out = model(x)
            loss = loss_fn(out, target)
            running_loss += loss.item()
        
        o_fair = torch.zeros(size=(a.size()[0], 1)).cuda()
        fair = torch.zeros(size=(a.size()[0], 1)).cuda()
        for __ in range(100):
            ux = data[:, 1 : d + 1].detach().clone().cuda()
            cux = data[:, 1 : d + 1].detach().clone().cuda()
            uy = torch.Tensor(np.random.uniform(0, 1, size=(ux.size()[0], 1))).cuda()
            cuy = uy.detach().clone()
            
            x_check = get_x(ux, 3 - a, simdata.wux, simdata.wax)
            y_check = get_y(uy, x_check, simdata.wxy, simdata.wuy)
            x = get_x(ux, a, simdata.wux, simdata.wax)
            y = get_y(uy, x, simdata.wxy, simdata.wuy)
            o_fair += torch.abs(y - y_check).detach().clone()
            
            if i == 0:
                Y_current.append(y[0].item())
                cY_current.append(y_check[0].item())
            
            ux.requires_grad = True
            x = get_x(ux, a, simdata.wux, simdata.wax)
            out = model(x)
            gradient_dummy = torch.ones_like(out)
            out.backward(gradient=gradient_dummy)
            grad_ux = ux.grad
            f_ux = ux + eta * grad_ux
            f_x = get_x(f_ux, a, simdata.wux, simdata.wax)
            f_y = get_y(uy, f_x, simdata.wxy, simdata.wuy)
            
            cux.requires_grad = True
            cx = get_x(cux, 3 - a, simdata.wux, simdata.wax)
            cout = model(cx)
            cgradient_dummy = torch.ones_like(cout)
            cout.backward(gradient=cgradient_dummy)
            grad_cux = cux.grad
            f_cux = cux + eta * grad_cux
            f_cx = get_x(f_cux, 3 - a, simdata.wux, simdata.wax)
            f_cy = get_y(cuy, f_cx, simdata.wxy, simdata.wuy)
            
            fair += torch.abs(f_y - f_cy).detach().clone()
            
            if i == 0:
                Y_prime.append(f_y[0].item())
                cY_prime.append(f_cy[0].item())
        
        o_fair = o_fair / 100
        fair = fair / 100
        
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

def UFexperiment(loaders, total_epochs, eta, simdata):
    train_loader, valid_loader, test_loader = loaders[0], loaders[1], loaders[2]
    model = UFdecision(input_dim=simdata.d)
    model.cuda()
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.99))
    UFtrain(train_loader, valid_loader, model, loss_fn, optimizer, total_epochs, eta, simdata)
    original_unfairness, unfairness, test_loss = UFtest(test_loader, model, loss_fn, eta, simdata)
    print("UF original_unfairness = {}, unfairness = {}, test_loss = {}".format(original_unfairness, unfairness, test_loss))      
    return original_unfairness, unfairness, test_loss
      
class CFdecision(nn.Module):
    def __init__(self, input_dim):
        super(CFdecision, self).__init__()
        self.layer = nn.Linear(input_dim, 1)
    
    def forward(self, x):
        out = self.layer(x)
        return out

def CFtrain(loader, vloader, model, loss_fn, optimizer, total_epochs, eta, simdata):
    losses = []
    valid_losses = []
    
    d = simdata.d
    
    for __ in range(total_epochs):
        model.train()
        running_loss = 0
        for __, data in enumerate(loader):
            optimizer.zero_grad()
            ux = data[:, 1 : d + 1].detach().clone().cuda()
            uy = torch.Tensor(np.random.uniform(0, 1, size=(ux.size()[0], 1))).detach().clone().cuda()
            a = data[:, 0].unsqueeze(1).detach().clone().cuda()
            
            x = get_x(ux, a, simdata.wux, simdata.wax)
            y = get_y(uy, x, simdata.wxy, simdata.wuy)
            
            u = torch.cat([ux, uy], dim=1).cuda()
            out = model(u)
            loss = loss_fn(out, y)
            loss.backward()
            running_loss += loss.item()
            optimizer.step()
        losses.append(running_loss / len(loader))
        
        model.eval()
        running_loss = 0
        for __, data in enumerate(vloader):
            with torch.no_grad():
                ux = data[:, 1 : d + 1].detach().clone().cuda()
                uy = data[:, 2 * d + 1].unsqueeze(1).detach().clone().cuda()
                a = data[:, 0].unsqueeze(1).detach().clone().cuda()
                
                x = get_x(ux, a, simdata.wux, simdata.wax)
                y = get_y(uy, x, simdata.wxy, simdata.wuy)
                
                u = torch.cat([ux, uy], dim=1).cuda()
                out = model(u)
                loss = loss_fn(out, y)
                running_loss += loss.item()
        valid_losses.append(running_loss / len(loader))
    
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig("images/CF_train_loss.png")
    plt.close()
    plt.plot(valid_losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig("images/CF_valid_loss.png")
    plt.close()

def CFtest(loader, model, loss_fn, eta, simdata):
    model.eval()
    
    d = simdata.d
    
    original_unfairs = []
    unfairs = []
    running_loss = 0
    
    Y_prime = []
    cY_prime = []
    
    for i, data in enumerate(loader):
        a = data[:, 0].unsqueeze(1).detach().clone().cuda()
        ux = data[:, 1 : d + 1].detach().clone().cuda()
        target = data[:, 2 * d + 2].unsqueeze(1).detach().clone().cuda()
        
        o_fair = torch.zeros(size=(a.size()[0], 1)).cuda()
        fair = torch.zeros(size=(a.size()[0], 1)).cuda()
        for __ in range(100):
            ux = data[:, 1 : d + 1].detach().clone().cuda()
            cux = data[:, 1 : d + 1].detach().clone().cuda()
            uy = torch.Tensor(np.random.uniform(0, 1, size=(ux.size()[0], 1))).cuda()
            cuy = uy.detach().clone()
            
            with torch.no_grad():
                u = torch.cat([ux, uy], dim=1)
                out = model(u)
                loss = loss_fn(out, target)
                running_loss += loss.item()
                
                x = get_x(ux, a, simdata.wux, simdata.wax)
                y = get_y(uy, x, simdata.wxy, simdata.wuy)
                x_check = get_x(ux, 3 - a, simdata.wux, simdata.wax)
                y_check = get_y(uy, x_check, simdata.wxy, simdata.wuy)
                
                o_fair += torch.abs(y - y_check).detach().clone()
            
            u = torch.cat([ux, uy], dim=1)
            u.requires_grad = True
            out = model(u)
            gradient_dummy = torch.ones_like(out)
            out.backward(gradient_dummy)
            grad_ux = u.grad[:, 0 : -1]
            grad_uy = u.grad[:, -1].unsqueeze(1)
            f_ux = ux + eta * grad_ux
            f_uy = uy + eta * grad_uy
            f_x = get_x(f_ux, a, simdata.wux, simdata.wax)
            f_y = get_y(f_uy, f_x, simdata.wxy, simdata.wuy)
            
            cu = torch.cat([cux, cuy], dim=1)
            cu.requires_grad = True
            cout = model(cu)
            cgradient_dummy = torch.ones_like(cout)
            cout.backward(gradient=cgradient_dummy)
            grad_cux = cu.grad[:, 0 : -1]
            grad_cuy = cu.grad[:, -1].unsqueeze(1)
            f_cux = cux + eta * grad_cux
            f_cuy = cuy + eta * grad_cuy
            f_cx = get_x(f_cux, 3 - a, simdata.wux, simdata.wax)
            f_cy = get_y(f_cuy, f_cx, simdata.wxy, simdata.wuy)
            
            fair += torch.abs(f_y - f_cy).detach().clone()
            
            if i == 0:
                Y_prime.append(f_y[0].item())
                cY_prime.append(f_cy[0].item())
        
        o_fair = o_fair / 100
        fair = fair / 100
        
        original_unfairs.extend([o_fair[i].item() for i in range(o_fair.size()[0])])
        unfairs.extend([fair[i].item() for i in range(fair.size()[0])])
    
    original_unfairness = np.mean(original_unfairs)
    unfairness = np.mean(unfairs)
    loss = running_loss / (100 * len(loader))
    
    with open("CF_Y.pkl", "wb") as f:
        pickle.dump(Y_prime, f)
    with open("CF_cY.pkl", "wb") as f:
        pickle.dump(cY_prime, f)
    
    return original_unfairness, unfairness, loss

def CFexperiment(loaders, total_epochs, eta, simdata):
    train_loader, valid_loader, test_loader = loaders[0], loaders[1], loaders[2]
    model = CFdecision(input_dim=simdata.d + 1)
    model.cuda()
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.99))
    CFtrain(train_loader, valid_loader, model, loss_fn, optimizer, total_epochs, eta, simdata)
    original_unfairness, unfairness, test_loss = CFtest(test_loader, model, loss_fn, eta, simdata)
    print("CF original_unfairness = {}, unfairness = {}, test_loss = {}".format(original_unfairness, unfairness, test_loss))      
    return original_unfairness, unfairness, test_loss
            
            
class DFdecision(nn.Module):
    def __init__(self, input_dim):
        super(DFdecision, self).__init__()
        self.w1 = torch.Tensor([0]).cuda()
        self.w2 = nn.Parameter(torch.randn(1))
        self.w3 = nn.Parameter(torch.randn(1))
        
        self.layer = nn.Linear(input_dim, 1)
    
    def forward(self, cy, u):
        h1 = self.w1 * cy ** 2 + self.w2 * cy + self.w3
        #h1 = self.w1 * cy ** 2
        #h2 = self.layer(u)
        out = h1
        return out

def DFtrain(loader, vloader, model, loss_fn, optimizer, total_epochs, eta, simdata):
    losses = []
    valid_losses = []
    
    d = simdata.d
    
    for __ in range(total_epochs):
        model.train()
        running_loss = 0
        for __, data in enumerate(loader):
            optimizer.zero_grad()
            ux = data[:, 1 : d + 1].detach().clone().cuda()
            uy = torch.Tensor(np.random.uniform(0, 1, size=(ux.size()[0], 1))).detach().clone().cuda()
            a = data[:, 0].unsqueeze(1).detach().clone().cuda()
            
            x = get_x(ux, a, simdata.wux, simdata.wax)
            y = get_y(uy, x, simdata.wxy, simdata.wuy)
            x_check = get_x(ux, 3 - a, simdata.wux, simdata.wax)
            y_check = get_y(uy, x_check, simdata.wxy, simdata.wuy)
            
            u = torch.cat([ux, uy], dim=1).cuda()
            out = model(y_check, u)
            loss = loss_fn(out, y)
            loss.backward()
            running_loss += loss.item()
            optimizer.step()
        losses.append(running_loss / len(loader))
        
        model.eval()
        running_loss = 0
        for __, data in enumerate(vloader):
            with torch.no_grad():
                ux = data[:, 1 : d + 1].detach().clone().cuda()
                uy = data[:, 2 * d + 1].unsqueeze(1).detach().clone().cuda()
                a = data[:, 0].unsqueeze(1).detach().clone().cuda()
                
                x = get_x(ux, a, simdata.wux, simdata.wax)
                y = get_y(uy, x, simdata.wxy, simdata.wuy)
                x_check = get_x(ux, 3 - a, simdata.wux, simdata.wax)
                y_check = get_y(uy, x_check, simdata.wxy, simdata.wuy)
                
                u = torch.cat([ux, uy], dim=1).cuda()
                out = model(y_check, u)
                loss = loss_fn(out, y)
                running_loss += loss.item()
        valid_losses.append(running_loss / len(vloader))
        
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig("images/DF_train_loss.png")
    plt.close()
    plt.plot(valid_losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig("images/DF_valid_loss.png")
    plt.close()

def DFtest(loader, model, loss_fn, eta, ratio, simdata):
    model.eval()
    
    d = simdata.d
    
    original_unfairs = []
    unfairs = []
    running_loss = 0
    
    Y_prime = []
    cY_prime = []
    
    for i, data in enumerate(loader):
        a = data[:, 0].unsqueeze(1).detach().clone().cuda()
        ux = data[:, 1 : d + 1].detach().clone().cuda()
        target = data[:, 2 * d + 2].unsqueeze(1).detach().clone().cuda()
        
        o_fair = torch.zeros(size=(a.size()[0], 1)).cuda()
        fair = torch.zeros(size=(a.size()[0], 1)).cuda()
        for __ in range(100):
            ux = data[:, 1 : d + 1].detach().clone().cuda()
            cux = data[:, 1 : d + 1].detach().clone().cuda()
            uy = torch.Tensor(np.random.uniform(0, 1, size=(ux.size()[0], 1))).cuda()
            cuy = uy.detach().clone()
            
            with torch.no_grad():
                u = torch.cat([ux, uy], dim=1)
                x_check = get_x(ux, 3 - a, simdata.wux, simdata.wax)
                y_check = get_y(uy, x_check, simdata.wxy, simdata.wuy)
                out = model(y_check, u)
                loss = loss_fn(out, target)
                running_loss += loss.item()
                
                x = get_x(ux, a, simdata.wux, simdata.wax)
                y = get_y(uy, x, simdata.wxy, simdata.wuy)
                
                o_fair += torch.abs(y - y_check).detach().clone() 
            
            
            u = torch.cat([ux, uy], dim=1)
            u.requires_grad = True
            x_check = get_x(u[:, 0 : -1], 3 - a, simdata.wux, simdata.wax)
            y_check = get_y(u[:, -1].unsqueeze(1), x_check, simdata.wxy, simdata.wuy)
            out = model(y_check, u)
            graient_dummy = torch.ones_like(out)
            out.backward(gradient=graient_dummy)
            grad_ux = u.grad[:, 0 : -1]
            grad_uy = u.grad[:, -1].unsqueeze(1)
            f_ux = ux + eta * grad_ux
            f_uy = uy + eta * grad_uy
            f_x = get_x(f_ux, a, simdata.wux, simdata.wax)
            f_y = get_y(f_uy, f_x, simdata.wxy, simdata.wuy)
            
            cu = torch.cat([cux, cuy], dim=1)
            cu.requires_grad = True
            cx_check = get_x(cu[:, 0 : -1], a, simdata.wux, simdata.wax)
            cy_check = get_y(cu[:, -1].unsqueeze(1), cx_check, simdata.wxy, simdata.wuy)
            cout = model(cy_check, cu)
            cgradient_dummy = torch.ones_like(cout)
            cout.backward(gradient=cgradient_dummy)
            grad_cux = cu.grad[:, 0 : -1]
            grad_cuy = cu.grad[:, -1].unsqueeze(1)
            f_cux = cux + eta * grad_cux
            f_cuy = cuy + eta * grad_cuy
            f_cx = get_x(f_cux, 3 - a, simdata.wux, simdata.wax)
            f_cy = get_y(f_cuy, f_cx, simdata.wxy, simdata.wuy)
            
            fair += torch.abs(f_y - f_cy).detach().clone()
            
            if i == 0:
                Y_prime.append(f_y[0].item())
                cY_prime.append(f_cy[0].item())
        
        o_fair = o_fair / 100
        fair = fair / 100
        
        original_unfairs.extend([o_fair[i].item() for i in range(o_fair.size()[0])])
        unfairs.extend([fair[i].item() for i in range(fair.size()[0])])
    
    original_unfairness = np.mean(original_unfairs)
    unfairness = np.mean(unfairs)
    loss = running_loss / (100 * len(loader))
    
    with open("DF_Y_{}.pkl".format(ratio), "wb") as f:
        pickle.dump(Y_prime, f)
    with open("DF_cY_{}.pkl".format(ratio), "wb") as f:
        pickle.dump(cY_prime, f)
    
    return original_unfairness, unfairness, loss

def DFexperiment(loaders, total_epochs, eta, ratio, simdata):
    train_loader, valid_loader, test_loader = loaders[0], loaders[1], loaders[2]
    model = DFdecision(input_dim=simdata.d + 1)
    model.cuda()
    #model.w1 = torch.tensor(1 / (2 * ratio * eta * (np.sum(simdata.wxy * simdata.wxy * simdata.wux * simdata.wux) + np.sum(simdata.wuy * simdata.wuy)))).cuda()
    model.w1 = torch.tensor(1 / (2 * ratio * eta * (2 * np.sum(simdata.wxy * simdata.wxy * simdata.wux * simdata.wux) + np.sum(simdata.wuy * simdata.wuy)))).cuda()
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.99))
    DFtrain(train_loader, valid_loader, model, loss_fn, optimizer, total_epochs, eta, simdata)
    original_unfairness, unfairness, test_loss = DFtest(test_loader, model, loss_fn, eta, ratio, simdata)
    print("DF original_unfairness = {}, unfairness = {}, test_loss = {}".format(original_unfairness, unfairness, test_loss))
    return original_unfairness, unfairness, test_loss
    
def drawDensity(simdata):
    simdata.splitData(seed=47)
    for i in range(4):
        train_set = simDataset(simdata, type="train")
        valid_set = simDataset(simdata, type="valid")
        test_set = simDataset(simdata, type="test")
        train_loader = DataLoader(train_set, batch_size=128, shuffle=True)
        valid_loader = DataLoader(valid_set, batch_size=128, shuffle=False)
        test_loader = DataLoader(test_set, batch_size=128, shuffle=False)
        loaders = [train_loader, valid_loader, test_loader]
        
        total_epochs = 2000
        eta = 10
        if i == 0:
            UFexperiment(loaders, total_epochs, eta, simdata)
        elif i == 1:
            CFexperiment(loaders, total_epochs, eta, simdata)
        elif i == 2:
            DFexperiment(loaders, total_epochs, eta, 1, simdata)
        else:
            DFexperiment(loaders, total_epochs, eta, 2, simdata)
                
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
    plt.rcParams['text.usetex'] = True

    def plot_single_graph(i, factual_data, counter_data, ax, title):
        kde1 = gaussian_kde(factual_data)
        kde2 = gaussian_kde(counter_data)
        
        if i == 0:
            x = np.linspace(-2, 4, 100)
        elif i == 1:
            x = np.linspace(3, 8, 100)
        elif i == 2:
            x = np.linspace(9.5, 12.5, 100)
        elif i == 3:
            x = np.linspace(8, 13, 100)
        elif i == 4:
            x = np.linspace(8, 13, 100)
        density1 = kde1(x)
        density2 = kde2(x)

        ax.plot(x, density1, color="b", label="factual")
        ax.plot(x, density2, color="r", label="counterfactual")

        ax.fill_between(x, density1, color='b', alpha=0.2)
        ax.fill_between(x, density2, color='r', alpha=0.2)
        
        #ax.legend()
        if i != 0:
            ax.set_xlabel("$Y'$")
        else:
            ax.set_xlabel("$Y$")
        if i == 0:
            ax.set_ylabel("density")
        ax.set_title(title)

    fig, axs = plt.subplots(1, 5, figsize=(50, 10))


    #titles = ["Baseline", "CA", "ICA", "CE", "CR", "Ours"]
    titles = ["Current", "UF", "CF", r"DF($p_{1}=\frac{T}{4}$)", r"DF($p_{1}=\frac{T}{2}$)"]
    for i, title in enumerate(titles):
        plot_single_graph(i, factuals[i], counters[i], axs[i], title)
    #plt.subplots_adjust(left=None, bottom=0.15, right=None, top=0.85, wspace=0.3, hspace=0.8)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
    lineA = mlines.Line2D([], [], color='b', label='factual')
    lineB = mlines.Line2D([], [], color='r', label='counter\nfactual')
    # Create a legend for the whole figure
    fig.legend(handles=[lineA, lineB], loc = "center right", ncol=1, framealpha=1)
    
    #plt.tight_layout()
    plt.show()
    plt.savefig("density.pdf")
    #plt.savefig("density_cvae.pdf")

def compute_table(simdata):
    o_fairnesses = np.zeros((5, 3))
    fairnesses = np.zeros((5, 3))
    losses = np.zeros((5, 3))
    for num in range(5):
        simdata.splitData(seed=42 + num)
        for i in range(3):
            train_set = simDataset(simdata, type="train")
            valid_set = simDataset(simdata, type="valid")
            test_set = simDataset(simdata, type="test")
            train_loader = DataLoader(train_set, batch_size=128, shuffle=True)
            valid_loader = DataLoader(valid_set, batch_size=128, shuffle=False)
            test_loader = DataLoader(test_set, batch_size=128, shuffle=False)
            loaders = [train_loader, valid_loader, test_loader]
            
            total_epochs = 2000
            eta = 10
            if i == 0:
                original_unfairness, fairness, loss = UFexperiment(loaders, total_epochs, eta, simdata)
            elif i == 1:
                original_unfairness, fairness, loss = CFexperiment(loaders, total_epochs, eta, simdata)
            elif i == 2:
                original_unfairness, fairness, loss = DFexperiment(loaders, total_epochs, eta, 1, simdata)
            o_fairnesses[num, i], fairnesses[num, i], losses[num, i] = original_unfairness, fairness, loss
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

def plot_trade_off(simdata):
    unfairs = np.zeros((3, 6))
    losses = np.zeros((3, 6))

    simdata.splitData(seed=47)
    
    etas = [10, 5, 1]
    ratios = [1, 2, 4, 16, 32, 256]
    
    for i, eta in enumerate(etas):
        for j, ratio in enumerate(ratios):
            train_set = simDataset(simdata, type="train")
            valid_set = simDataset(simdata, type="valid")
            test_set = simDataset(simdata, type="test")
            train_loader = DataLoader(train_set, batch_size=128, shuffle=True)
            valid_loader = DataLoader(valid_set, batch_size=128, shuffle=False)
            test_loader = DataLoader(test_set, batch_size=128, shuffle=False)
            loaders = [train_loader, valid_loader, test_loader]
            total_epochs = 2000
            __, unfairness, loss = DFexperiment(loaders, total_epochs, eta, ratio, simdata)
            unfairs[i, j] = unfairness
            losses[i, j] = loss
    
    font = {"family": "normal", "size": 13}
    matplotlib.rc("font", **font)

    fig = plt.figure(figsize=(5, 4.4))
    p1, = plt.plot(losses[0], unfairs[0], "r*-", label="$\eta=10$", linewidth=2)
    p2, = plt.plot(losses[1], unfairs[1], "c>:", label="$\eta=5$", linewidth=2)
    p3, = plt.plot(losses[2], unfairs[2], "y^-", label="$\eta=1$", linewidth=2)
    #p4, = plt.plot(losses[3], unfairs[3], "bx--", label="$\eta=0.5$", linewidth=2)
    plt.legend()

    plt.xlabel("MSE")
    plt.ylabel("AFCE")
    plt.grid()
    plt.show()
    fig.savefig("trade_off_curves.png")
                      
def main():
    simdata = simData(N=1000, d=10)
    compute_table(simdata)
    drawDensity(simdata)
    plot_trade_off(simdata)
            
if __name__ == "__main__":
    main()