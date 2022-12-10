#!/usr/bin/env python
# coding: utf-8

# In[11]:


import torch
from torch import nn
from torch import optim
from torch.utils.data import Dataset


# In[42]:


#data preparation
class TitanicDataset(Dataset):
    def __init__(self,csvpath, mode = 'train'):
        self.mode = mode
        df = pd.read_csv(csvpath)
        #le = LabelEncoder()        
        if self.mode == 'train':
            df = df.dropna()
            self.inp = df.iloc[:,0:4].values
            self.oup = df.iloc[:,-1].values.reshape(871,1)
        else:
            self.inp = df.values
    def __len__(self):
        return len(self.inp)
    def __getitem__(self, idx):
        if self.mode == 'train':
            inpt  = torch.Tensor(self.inp[idx])
            oupt  = torch.Tensor(self.oup[idx])
            return { 'inp': inpt,
                     'oup': oupt,
            }
        else:
            inpt = torch.Tensor(self.inp[idx])
            return { 'inp': inpt
            }
## Initialize the DataSet
train = TitanicDataset('train.csv')
## Load the Dataset
data_train = DataLoader(dataset = train, batch_size = 16, shuffle =False)

test = TitanicDataset('test.csv')
## Load the Dataset
data_test = DataLoader(dataset = test, batch_size = 16, shuffle =False)


# In[44]:


#Architecture
num_in = 4
hidden_dim = 5
num_ou = 1

#relu
class NeuralNetwork(nn.Module):
    def __init__(self, num_in, hidden_dim, num_ou):
        super(NeuralNetwork, self).__init__()
        self.layer_1 = nn.Linear(num_in, hidden_dim)
        self.layer_2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer_3 = nn.Linear(hidden_dim, num_ou)
        nn.init.kaiming_uniform_(self.layer_1.weight, nonlinearity="relu")#"he" 
        nn.init.kaiming_uniform_(self.layer_2.weight, nonlinearity="relu")#"he" 
        nn.init.kaiming_uniform_(self.layer_3.weight, nonlinearity="relu")#"he" 
       
    def forward(self, x):
        x = torch.nn.functional.relu(self.layer_1(x))
        x = torch.nn.functional.relu(self.layer_2(x))
        x = torch.nn.functional.relu(self.layer_3(x))
        return x
       
model = NeuralNetwork(num_in, hidden_dim, num_ou)
print(model)


# In[28]:


#tanh
class NeuralNetwork(nn.Module):
    def __init__(self, num_in, hidden_dim, num_ou):
        super(NeuralNetwork, self).__init__()
        self.layer_1 = nn.Linear(num_in, hidden_dim)
        self.layer_2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer_3 = nn.Linear(hidden_dim, num_ou)
        nn.init.xavier_uniform_(self.layer_1.weight)
        nn.init.xavier_uniform_(self.layer_2.weight)
        nn.init.xavier_uniform_(self.layer_3.weight)

    def forward(self, x):
        x = torch.nn.functional.Tanh(self.layer_1(x))
        x = torch.nn.functional.Tanh(self.layer_2(x))
        x = torch.nn.functional.Tanh(self.layer_3(x))
        return x
       
model = NeuralNetwork(num_in, hidden_dim, num_ou)
print(model)


# In[47]:


from torch.optim import Adam
criterion = nn.MSELoss()
EPOCHS = 200
optm = Adam(model.parameters(), lr = 0.001)


# In[48]:


def train(model, x, y, optimizer, criterion):
    model.zero_grad()
    output = model(x)
    loss =criterion(output,y)
    loss.backward()
    optimizer.step()
    return loss, output


# In[ ]:




