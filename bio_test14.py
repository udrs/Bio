# c = torch.ones([1824])
import torch
from torch.autograd import Variable
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import torch
import pandas as pd
import numpy as np
import gc
import torch.nn as nn
from dataset import TrainSet, ValSet
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt


c_val = torch.load('./weights/28_c_avg15.pt')
c_val = c_val.to(device)
c_val = Variable(c_val, requires_grad=True)

root_dir = '/media/lscsc/nas/yihan/bio/dataset/'

transform = transforms.Compose([
    transforms.ToTensor()])

valset = ValSet(root_dir, transform)
# valset = ValSet(root_dir, transform)
valloader = DataLoader(valset, batch_size=398, shuffle=True, num_workers=0)


for j, val_datalist in enumerate(valloader):
    
        native_f, f, u, transaction, Eem = val_datalist
        
        print("val:",len(Eem))
        
    
        # print(f'{epoch}_{j}')
        f = f.to(device)
        native_f = native_f.to(device)
        u = u.to(device)
        transaction = transaction.to(device)
        Eem = Eem.to(device)
        
        # transaction = sum(transaction)/len(transaction)
        # Eem avg_Eem computation
        # print("Eem",Eem)
        Eem_avg = torch.sum(Eem)/len(Eem)
        # print("Avg_Eem",Eem_avg)

        
        # Epm avg_Epm computation
        
        Epm = torch.zeros([len(Eem)])
        Epm = Epm.to(device)
        # print("Epm:",Epm.shape)
        # print("c:",c.shape)
        # print("u:",u.shape)
        # print("native_f:",native_f.shape)
        for i in range(len(Eem)):
            for j in range(1824):
                Epm[i] = Epm[i] + (c_val[j]*u[i][j]*native_f[i][j])
        # print("Emp:",Epm.shape)
        avg_Epm = abs(torch.sum(Epm)/len(Eem))
        # print("avg_Epm:",avg_Epm)
        
        
        
        
        # Gama computation
        up_sum = torch.tensor([0])
        down_sum_left = torch.tensor([0])
        down_sum_right = torch.tensor([0])
        
        up_sum = up_sum.to(device)
        down_sum_left = down_sum_left.to(device)
        down_sum_right = down_sum_right.to(device)
        
        avg_Epm = torch.as_tensor(avg_Epm, dtype=torch.float32)
        Eem_avg = torch.as_tensor(Eem_avg, dtype=torch.float32)
        # avg_Epm = torch.tensor(avg_Epm).to(device)
        # Eem_avg = torch.tensor(Eem_avg).to(device)
        
        for i in range(len(Eem)):
            up_sum = up_sum + ((Epm[i] - avg_Epm)*(Eem[i] - Eem_avg))
            down_sum_left = down_sum_left + ((Epm[i] - avg_Epm)*(Epm[i] - avg_Epm))
            down_sum_right = down_sum_right + ((Eem[i] - Eem_avg)*(Eem[i] - Eem_avg))
        Gama = up_sum/(torch.sqrt(down_sum_right)*torch.sqrt(down_sum_left))
        
        
        
        sumf = torch.sum(f,dim=1)
        avg_f = sumf/250  # [50, 1824]
        
        
        
        
        avg_fkk = torch.sum(avg_f,dim=1)/1824  # 50
        avg_fll = avg_fkk
        avg_avg_fkk = torch.sum(avg_fkk)/len(Eem)
        avg_avg_fll = torch.sum(avg_fll)/len(Eem)
        f_dif = torch.rand([len(Eem),1824])
        for i in range(len(Eem)):
            f_dif[i] = native_f[i] - avg_fkk[i]
        avg_f_dif = torch.sum(f_dif,dim=0)/len(Eem)
        avg_u = torch.sum(u , dim = 0)/len(Eem)
        






        theta_E = torch.zeros([1])
        theta_E = theta_E.to(device)
        for i in range(1824):
            theta_E = theta_E + abs(avg_f_dif[i]*avg_u[i]*c_val[i])
        print("theta_E:",theta_E)


        ######################  To Now  ###################



        fk_fl = torch.zeros([len(Eem),1824])
        fk_fl = fk_fl.to(device)
        for k in range(len(Eem)):
            for i in range(250):
                for j in range(250):
                    fk_fl[k] = fk_fl[k] + f[k][i]*f[k][j]

        fk_fl = fk_fl/(250*250)
        avg_fk_fl = torch.sum(fk_fl,dim=1)
        avg_avg_fk_fl = avg_fk_fl/1824
        lambda_target = torch.tensor([7]).to(device)
        avg_avg_avg_fk_fl = torch.sum(avg_avg_fk_fl)/len(Eem)

        delta_E = torch.zeros([1])
        delta_E = delta_E.to(device)



        for i in range(1824):
            for j in range(1824):
                delta_E = delta_E + c_val[j]*c_val[i]*avg_u[j]*avg_u[i]*(avg_avg_avg_fk_fl-avg_avg_fkk*avg_avg_fll)
        print("delta_E:",delta_E)

        scale = torch.tensor([7])
        scale = scale.to(device)
        
        lambda_E = torch.zeros([1])
        # print(transaction.is_cuda)
        # for i in range(1):
    
        lambda_E = (abs(theta_E*scale))/torch.sqrt(abs(delta_E)*torch.sqrt((transaction+scale))) 
        
        lambda_up = 0
        lambda_down = 0
        
        
        beta_minus = torch.tensor([-0.1])
        beta_minus = beta_minus.to(device)
        for i in range(len(lambda_E)):
            lambda_up = lambda_up + lambda_E[i]*torch.exp(beta_minus*lambda_E[i])
            lambda_down = lambda_down + torch.exp(beta_minus*lambda_E[i])
        
        avg_lambda_E = lambda_up/lambda_down
        
        Ro = (avg_lambda_E*Gama)
        print("lambda:",avg_lambda_E)
        print("gama:",Gama)
        print("RoL",Ro)
        # Ro_val = str(Ro)
        # Gama_val = str(Gama)
        # lambda_val = str(avg_lambda_E)
