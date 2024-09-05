import scipy.io
import random
import torch
import numpy as np
from torch import nn
import matplotlib.pylab as plt
class MLP(nn.Module):

    def __init__(self, nInput, nOutput):
        super(MLP, self).__init__()

        # two hidden layers
        self.layers = nn.Sequential(
            nn.Linear(nInput, 40),
            nn.ReLU(),
            nn.Linear(40, 60),
            nn.ReLU(),
            nn.Linear(60, nOutput)
        )

    def forward(self, x):
        x = self.layers(x)
        return x

# from google.colab import drive
# drive.mount('/content/drive/')

nInput = 2
nOutput = 2+2+2
model_a = MLP(nInput, nOutput)
model_b = MLP(nInput, nOutput)
model_a.load_state_dict(torch.load('model_a_1520.pt'))
model_b.load_state_dict(torch.load('model_b_1520.pt'))

train_data_a = scipy.io.loadmat('data_a_2_1520.mat')
a_train_db = torch.tensor(train_data_a['db'])
a_train_ctrl = torch.tensor(train_data_a['ctrl'])
a_means = torch.tensor(train_data_a['means'])
train_data_b = scipy.io.loadmat('data_b_2_1520.mat')
b_train_da = torch.tensor(train_data_b['da'])
b_train_ctrl = torch.tensor(train_data_b['ctrl'])
b_means = torch.tensor(train_data_b['means'])

FREQS = torch.tensor(train_data_b['freqs'])
MEAN_SPEED = torch.tensor(train_data_b['speeds'])


def getControl(da, db):
    mag_a = torch.sqrt(torch.dot(da,da))
    mag_b = torch.sqrt(torch.dot(db,db))
    if mag_a > mag_b:
        angle = torch.arctan2(da[1], da[0]) - torch.pi/2;
        A = torch.tensor([[torch.cos(-angle), -torch.sin(-angle)], [torch.sin(-angle), torch.cos(-angle)]]) / mag_a
        db = A @ db

        reflect = False
        if db[0] < 0:
            db[0] = -db[0]
            reflect = True

        ctrl = model_a(db.float()).double() + a_means
        if reflect:
            ctrl[:,2:4] = torch.arctan2(torch.sin(ctrl[:,2:4]), -torch.cos(ctrl[:,2:4]))

        ctrl[:,2:4] += angle
        ctrl[:,4:6] *= mag_a

        return ctrl
    else:
        angle = torch.arctan2(db[1], db[0]) - torch.pi/2;
        A = torch.tensor([[torch.cos(-angle), -torch.sin(-angle)], [torch.sin(-angle), torch.cos(-angle)]]) / mag_b
        da = A @ da

        reflect = False
        if da[0] < 0:
            da[0] = -da[0]
            reflect = True


        ctrl = model_b(da.float()).double() + b_means

        if reflect:
            ctrl[:,2:4] = torch.arctan2(torch.sin(ctrl[:,2:4]), -torch.cos(ctrl[:,2:4]))

        ctrl[:,2:4] += angle
        ctrl[:,4:6] *= mag_b

        return ctrl