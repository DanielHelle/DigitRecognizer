import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from torch import optim
import model as m


def main():
    df_train = pd.read_csv("train.csv")
    df_test = pd.read_csv("test.csv")
    df_train_l = df_train.pop("label")

    train_np = df_train.to_numpy()
    train_l_np = df_train_l.to_numpy()
    test_l_np = np.zeros(df_test.iloc[:,0].to_numpy().shape)
    test_np = df_test.to_numpy()

    train = torch.from_numpy(train_np)
    train_l = torch.from_numpy(train_l_np)
    test = torch.from_numpy(test_np)
    test_l = torch.from_numpy(test_l_np)

    train = train.view(-1,1,28,28)
    test = test.view(-1,1,28,28)

    train_l.long()


  
    




    print(train.type)
    



    
    train_set = torch.utils.data.TensorDataset(train,train_l)
    test_set = torch.utils.data.TensorDataset(test,test_l)
    train_loader = torch.utils.data.DataLoader(train_set, shuffle = True)
    test_loader = torch.utils.data.DataLoader(test_set,shuffle = True)
    net = m.Net()
    m.fit(net,train_loader)



    return 0






if __name__ == "__main__":
        main()