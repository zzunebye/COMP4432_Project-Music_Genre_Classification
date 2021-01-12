import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from matplotlib import pyplot as plt

class Conv2d_net(nn.Module):
    def __init__(self):
        super(Conv2d_net, self).__init__() # 10*10
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2) # 6@10*10 # 한개 채널의 2d 이미지를 3x3 커널로 6개의 채널을 출력
        self.pool = nn.MaxPool2d(2, 2) # 6@5*5
        self.conv2 = nn.Conv2d(6, 10, 2, padding=1) # 16@ 6*6

        self.fc1 = nn.Linear(10 * 6 * 6, 84) 
        self.fc2 = nn.Linear(84, 36)
        self.fc3 = nn.Linear(36, 3)

        self.drop_3 = nn.Dropout(0.3)
        self.conv1_bn = nn.BatchNorm2d(6)
        self.conv2_bn = nn.BatchNorm2d(10)

    def forward(self, x):
        x = self.pool(F.elu(self.conv1_bn(self.conv1(x))))
        x = F.elu(self.conv2_bn(self.conv2(x)))
        x = x.view(-1, 10 * 6 * 6)
        x = self.drop_3(F.elu((self.fc1(x))))
        x = self.drop_3(F.elu(self.fc2(x)))
        x = self.fc3(x)
        return x

print('\n***********************************\n\nLoading the model and the data...\n')

FILE_PATH = './data_test_task2.npy'

data_test_task2 = np.load(FILE_PATH)

tensor_x = torch.Tensor(data_test_task2)
tensor_x = tensor_x.unsqueeze(1)

criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(task2_model.parameters(), lr=0.002)

PATH = "./state_dict_task2_model.pt"

val_model = Conv2d_net()
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

np.save('label_pred_task2.npy', yields)