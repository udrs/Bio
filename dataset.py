from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class TrainSet(Dataset):

    def __init__(self, root_dir, transform=None):

        self.root_dir = root_dir
        
        self.decoy_root_dir = os.path.join(self.root_dir,'Add','decoy')
        self.native_root_dir = os.path.join(self.root_dir,'Add','native')
        self.U_root_dir = os.path.join(self.root_dir,'U','nativepotential_3.dat')
        # print("self.decoy_root_dir:",self.decoy_root_dir)
        # print("self.native_root_dir:",self.native_root_dir)
        # self.alpha = "/media/lscsc/nas/yihan/bio/alpha/Torsion2011.dat"
        
        self.decoy_f = os.listdir(self.decoy_root_dir)
        self.native_f = os.listdir(self.native_root_dir)
        # print(self.decoy_f)

    def __len__(self):
        return len(self.decoy_f)


    def __getitem__(self, idx):

        decoy_f_name = self.decoy_f[idx]
        # print("decoy_f_name:",decoy_f_name[:4])
        native_f_name = f'{decoy_f_name[:4]}_nativef.dat'
        U_name = self.U_root_dir

        decoy_f_path = os.path.join(self.root_dir, 'Add/decoy/', decoy_f_name)
        native_f_path = os.path.join(self.root_dir, 'Add/native/', native_f_name)
        # print(decoy_f_path)

        
        
        
        native_f_list = []
        nf = pd.read_csv(native_f_path, header=None) 
        for k in range(nf.shape[0]): 
                tmp = [i for i in nf[0][k].split(" ") if i != ""]
                native_f_list.append([float(i) for i in tmp])
        native_f_list = np.array(native_f_list)
        native_f = np.reshape(native_f_list, (-1))
        
        
        
        
        decoy = []
        decoy_group = []
        with open(decoy_f_path) as file:
            for ln in file: 
                ln = ln.strip("\r\t\n ") 
                if ln.find("decoy") >= 0: 
                    decoy.append(decoy_group)
                    decoy_group =[]
                    continue
                decoy_group.append([float(i) for i in ln.split(" ") if i!= ""])
        decoy.append(decoy_group) 
        f = np.array(decoy[1:])
        f = np.reshape(f, (250,-1)) 
        
        
        
        
        
        
        
        with open("/media/lscsc/nas/yihan/bio/dataset/real/RefinedSet2011.dat") as realfile:
                for ln in realfile:
                    if ln.find(decoy_f_name[:4]) >= 0:
                        E_em = eval(ln[19:23])
        
        
        
        with open("/media/lscsc/nas/yihan/bio/alpha/Torsion2011.dat") as realfile_alpha:
            for ln in realfile_alpha:
                ln = ln.strip("\r\t\n ") 
                # print(decoy[:4])
                if ln.find(decoy_f_name[:4]) >= 0:
                    # print(decoy[:4])
                    # print(ln)
                    transaction = eval(ln[8:9])
        
        
        
        
        
        
        
        U_new = []
        U_data = pd.read_csv(U_name, header=None) 
        for k in range(U_data.shape[0]): 
            tmp = [i for i in U_data[0][k].split(" ") if i != ""]
            U_new.append([float(i) for i in tmp[5:]])
        u = np.array(U_new)
        u = np.reshape(u, (-1)) #1824 = 16 x 114

        datalist = [native_f, f, u, transaction, E_em]
        return datalist




class ValSet(Dataset):

    def __init__(self, root_dir, transform=None):

        self.root_dir = root_dir
        self.decoy_root_dir = os.path.join(self.root_dir,'Add','decoy')
        self.native_root_dir = os.path.join(self.root_dir,'Add','native')
        self.U_root_dir = os.path.join(self.root_dir,'U','nativepotential_3.dat')
        # print("self.decoy_root_dir:",self.decoy_root_dir)
        # print("self.native_root_dir:",self.native_root_dir)
        # self.alpha = "/media/lscsc/nas/yihan/bio/alpha/Torsion2011.dat"
        
        self.decoy_f = os.listdir(self.decoy_root_dir)
        self.native_f = os.listdir(self.native_root_dir)
        # print(self.decoy_f)
        # self.transform = transform

        # self.imgs = os.listdir(root_dir)

    def __len__(self):
        return len(self.decoy_f)


    def __getitem__(self, idx):

        decoy_f_name = self.decoy_f[idx]
        # print("decoy_f_name:",decoy_f_name[:4])
        native_f_name = f'{decoy_f_name[:4]}_nativef.dat'
        U_name = self.U_root_dir

        decoy_f_path = os.path.join(self.root_dir, 'Add/decoy/', decoy_f_name)
        native_f_path = os.path.join(self.root_dir, 'Add/native/', native_f_name)
        # print(decoy_f_path)

        
        
        
        native_f_list = []
        nf = pd.read_csv(native_f_path, header=None) 
        for k in range(nf.shape[0]): 
                tmp = [i for i in nf[0][k].split(" ") if i != ""]
                native_f_list.append([float(i) for i in tmp])
        native_f_list = np.array(native_f_list)
        native_f = np.reshape(native_f_list, (-1))
        
        
        
        
        decoy = []
        decoy_group = []
        with open(decoy_f_path) as file:
            for ln in file: 
                ln = ln.strip("\r\t\n ") 
                if ln.find("decoy") >= 0: 
                    decoy.append(decoy_group)
                    decoy_group =[]
                    continue
                decoy_group.append([float(i) for i in ln.split(" ") if i!= ""])
        decoy.append(decoy_group) 
        f = np.array(decoy[1:])
        f = np.reshape(f, (250,-1)) 
        
        
        
        
        
        
        
        with open("/media/lscsc/nas/yihan/bio/dataset/real/RefinedSet2011.dat") as realfile:
                for ln in realfile:
                    if ln.find(decoy_f_name[:4]) >= 0:
                        E_em = eval(ln[19:23])
        
        
        
        with open("/media/lscsc/nas/yihan/bio/alpha/Torsion2011.dat") as realfile_alpha:
            for ln in realfile_alpha:
                ln = ln.strip("\r\t\n ") 
                # print(decoy[:4])
                if ln.find(decoy_f_name[:4]) >= 0:
                    # print(decoy[:4])
                    # print(ln)
                    transaction = eval(ln[8:9])
        
        
        
        
        
        
        
        U_new = []
        U_data = pd.read_csv(U_name, header=None) 
        for k in range(U_data.shape[0]): 
            tmp = [i for i in U_data[0][k].split(" ") if i != ""]
            U_new.append([float(i) for i in tmp[5:]])
        u = np.array(U_new)
        u = np.reshape(u, (-1)) #1824 = 16 x 114
        # image = Image.open(img_path)

        datalist = [native_f, f, u, transaction, E_em]
        return datalist




def test_dataset():

    root_dir = '/media/lscsc/nas/yihan/bio/dataset/'

    transform = transforms.Compose([
        transforms.ToTensor()])

    valset = ValSet(root_dir, transform)

    print(len(valset))

    valloader = DataLoader(valset, batch_size=48, shuffle=True, num_workers=0)

    for i, datalist in enumerate(valloader):
        print("1")
        # print("decoy_shape:",datalist[0].shape)

        
# test_dataset()