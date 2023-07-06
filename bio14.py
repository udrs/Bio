import torch
import pandas as pd
import numpy as np
from torch.autograd import Variable
import gc
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import torch
import torch.nn as nn
from dataset import TrainSet, ValSet
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt




loss_list = []
gama_list = []
lambda_list = []
def draw_loss(Loss_list,epoch):
    plt.cla()
    x1 = range(1, epoch+1)
    # print(x1)
    y1 = Loss_list
    # y1 = y1.cpu()
    y1 = torch.tensor(y1, device='cpu')
    # inputs = torch.tensor(inputs, device = 'cpu')
    # print(y1)
    plt.title('Train loss vs. epoches', fontsize=20)
    plt.plot(x1, y1, '.-')
    plt.xlabel('epoches', fontsize=20)
    plt.ylabel('Lambda', fontsize=20)
    plt.grid()
    plt.savefig("./Train_loss.png")
    plt.savefig("./Train_loss.png")
    plt.show()



c = torch.ones([1824])
# c = torch.load('./11_c_avg15.pt')
c = c.to(device)
c = Variable(c, requires_grad=True)







root_dir = '/media/lscsc/nas/yihan/bio/dataset/'

transform = transforms.Compose([
    transforms.ToTensor()])

trainset = TrainSet(root_dir, transform)
# valset = ValSet(root_dir, transform)
trainloader = DataLoader(trainset, batch_size=99, shuffle=True, num_workers=6)
# valloader = DataLoader(valset, batch_size=48, shuffle=True, num_workers=0)
# lr = 50000
epoch = 0
# optimizer = torch.optim.Adam([c], lr=0.5)
optimizer = torch.optim.Adam([c], lr=1)
decayRate = 0.96
my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=decayRate)




for epoch in range(160):
    torch.cuda.empty_cache()
    epoch = epoch + 1
  
    # optimizer = torch.optim.Adam([c], lr=0.1,weight_decay=1e-2)

    for i, datalist in enumerate(trainloader):
        
        native_f, f, u, transaction, Eem = datalist
        print(len(Eem))
        
        
       
        print(f'{epoch}_{i}')
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
                Epm[i] = Epm[i] + (c[j]*u[i][j]*native_f[i][j])
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
            theta_E = theta_E + abs(avg_f_dif[i]*avg_u[i]*c[i])
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
                delta_E = delta_E + c[j]*c[i]*avg_u[j]*avg_u[i]*(avg_avg_avg_fk_fl-avg_avg_fkk*avg_avg_fll)
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

        optimizer.zero_grad()
        

        gama_target = torch.tensor([1])
        gama_target = gama_target.cuda()


        loss_fn =  torch.nn.L1Loss()
        loss_fn2 =  torch.nn.L1Loss()
        # if Gama <= 0:
        #     loss1 = loss_fn(Gama,gama_target)
        # else:
        loss1 = 10 * loss_fn(Gama,gama_target)


        loss1.backward()
        


        lambda_saved = str(avg_lambda_E)
        Ro_saved = str(Ro)

        Gamast = str(Gama)
        avg_lambda_E = str(avg_lambda_E)
        
        # if Gama > -0.5:
        print("lambda:",avg_lambda_E)
        print("Gama:",Gama)
        print("Ro:",Ro)
        # print("entropy(Gama,gama_target):",entropy(Gama,gama_target))
        # print("loss2:",loss2)
        #  print("loss1:",loss1)
        print("loss:",loss1)
        with open(f'/media/lscsc/nas/yihan/bio/bio47.txt', 'a', encoding='utf-8') as f_gama:
                f_gama.write('gama ' + Gamast + '\n')
                f_gama.close()
        # lambda_saved = lambda_saved.split('(')[0].split(' ')[1]
        with open(f'/media/lscsc/nas/yihan/bio/bio47.txt', 'a', encoding='utf-8') as f:
                f.write('lambda ' + lambda_saved + '\n')
                f.close()
        with open(f'/media/lscsc/nas/yihan/bio/bio47.txt', 'a', encoding='utf-8') as f:
                f.write('Ro ' + Ro_saved + '\n'+'\n')
                f.close()

        optimizer.step()

        my_lr_scheduler.step()
        print(epoch, my_lr_scheduler.get_lr()[0])
        # c.data = c.data - lr*c.grad.data
        torch.cuda.empty_cache()
        # c.grad.data.zero_()
    torch.save(c,f'./weights/{epoch}_c_avg15.pt')
    
    


  