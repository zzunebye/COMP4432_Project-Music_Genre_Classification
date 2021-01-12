import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np


class Conv1d_net(nn.Module):
    def __init__(self):
        super(Conv1d_net, self).__init__() # 1*20
        self.conv1 = nn.Conv1d(1, 5, 2, padding=1) # 6@1*20
        self.pool = nn.MaxPool1d(2) # 6@10
        self.fc1 = nn.Linear(50, 24)
        self.fc2 = nn.Linear(24, 8)
        self.fc3 = nn.Linear(8, 3)

        self.drop_3 = nn.Dropout(0.3)
        self.drop_4 = nn.Dropout(0.4)
        self.conv1_bn = nn.BatchNorm1d(5)

    def forward(self, x):
        x = self.pool(F.elu(self.conv1_bn(self.conv1(x))))
        x = x.view(-1, 50) 
        x = self.drop_4(F.elu(self.fc1(x)))
        x = self.drop_3(F.elu(self.fc2(x)))
        x = self.fc3(x)
        return x

print('\n***********************************\n\nLoading the model and the data...\n')

FILE_PATH = './data_test_task1.npy'

data_test_task1 = np.load(FILE_PATH)

tensor_x = torch.Tensor(data_test_task1)
tensor_x = tensor_x.unsqueeze(1)

criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(task2_model.parameters(), lr=0.002)

PATH = "./state_dict_task1_model.pt"

val_model = Conv1d_net()
val_model.load_state_dict(torch.load(PATH))
val_model.eval()

print("the model from {} has parameters of\n\n{}".format(PATH,val_model.parameters))

correct = 0
outputs_list = []
total = 0

with torch.no_grad():
    tensor_x=tensor_x.float()
    tensor_x.shape
    outputs = val_model(tensor_x)
    _, predicted = torch.max(outputs.data, 1)
    outputs_list.append(predicted[:])
    total += tensor_x.size(0)
    
yields = torch.Tensor()
torch.cat(outputs_list, out=yields)
yields = yields.numpy()

print("\n___________________")
print("The predicted result from the input data {} is :".format(FILE_PATH))
print(yields,"\n")

answer = np.save('./label_pred_task1.npy',yields)
