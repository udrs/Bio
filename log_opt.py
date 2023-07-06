import matplotlib.pyplot as plt
import random
import torch
import gc

time_step = 0
def draw_loss(Loss_list,epoch,name):

    plt.cla()
    x1 = range(1, epoch+1)
    print(x1)
    y1 = Loss_list
    # y1 = y1.cpu()
    # y1 = torch.tensor(y1, device='cpu')
    # inputs = torch.tensor(inputs, device = 'cpu')
    print(y1)
    plt.title('Train loss vs. epoches', fontsize=20)
    plt.plot(x1, y1, '.-')
    plt.xlabel('epoches', fontsize=20)
    plt.ylabel(f'{name}', fontsize=20)
    plt.grid()
    plt.savefig(f"./47_{name}.png")
    # plt.savefig("./Train_loss.png")
    plt.show()






loss_list = []
gama_list = []
Ro_list = []
time_step = 0
sumloss = 0
gama_time_step = 0
gama_sumloss = 0
Ro_time_step = 0
Ro_sumloss = 0



# sum_time = 0
with open("/media/lscsc/nas/yihan/bio/bio47.txt") as file:
    for ln in file:
        # time_step = time_step + 1
        # if time_step>133:
        #     break
        # # sum_time = sum_time + 1
        # ln = ln.strip("\r\t\n")
        # # res = ln.split('lambda')[1]
    
        # print("lambda:",eval(ln[8:11]))
      
        # res_int = eval(ln[8:11])
        # loss_list.append(res_int)
        if ln.find("Ro") >= 0:
            

            res = ln.split('Ro')[1]

            print("Ro:",eval(res[9:15]))

            res_int = eval(res[9:15])
            
            # prob = random.randint(8,9)
            # if res_int<prob:
            Ro_time_step = Ro_time_step + 1
            Ro_sumloss = Ro_sumloss + res_int
            # if res_int < 9:
            if Ro_time_step == 4:
                Ro_list.append(Ro_sumloss/4)
                Ro_sumloss=0
                Ro_time_step=0
        


        if ln.find("gama") >= 0:
            

            res = ln.split('gama')[1]

            print("gama:",eval(res[9:15]))

            res_int = eval(res[9:15])
            
            # prob = random.randint(8,9)
            # if res_int<prob:
            gama_time_step = gama_time_step + 1
            gama_sumloss = gama_sumloss + res_int
            # if res_int < 9:
            if gama_time_step == 4:
                gama_list.append(gama_sumloss/4)
                gama_sumloss=0
                gama_time_step=0
        
        
        if ln.find("lambda") >= 0:
            

            res = ln.split('lambda')[1]

            print("lambda:",eval(res[9:15]))

            res_int = eval(res[9:15])
            
            # prob = random.randint(8,9)
            # if res_int<prob:
            time_step = time_step + 1
            sumloss = sumloss + res_int
            # if res_int < 9:
            if time_step == 4:
                loss_list.append(sumloss/4)
                sumloss=0
                time_step=0
        

name1 = "lambda"
name2 = "gama"
name3 = "Ro"
draw_loss(loss_list, 43, name1)
draw_loss(gama_list,43, name2)
draw_loss(Ro_list,43, name3)
# print(sum_time)
print(time_step)