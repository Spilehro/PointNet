from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F


class TNet(nn.Module):
    def __init__(self, k=64):
        super(TNet, self).__init__()
        self. k = k
        # Each layer has batchnorm and relu on it
        # conv 3 64
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 64, kernel_size = 3,stride = 1)
        self.bachnorm1 = nn.BatchNorm2d(num_features = 64)
        # conv 64 128
        self.conv2 = nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 3, stride = 1)
        self.bachnorm2 = nn.BatchNorm2d(num_features = 128)
        # conv 128 1024
        self.conv3 = nn.Conv2d(in_channels = 128, out_channels = 1024, kernel_size = 3, stride = 1)
        self.bachnorm3 = nn.BatchNorm2d(num_features = 1024)
        # max pool
        self.max_pool = nn.MaxPool2d(kernel_size = 3, stride = 1, padding = 1)
        # fc 1024 512
        self.fc1 = nn.Linear(in_features = 1024, out_features = 512)
        self.batchnorm_fc1 = nn.BatchNorm2d(num_features = 512)
        # fc 512 256
        self.fc2 = nn.Linear(in_features = 512, out_features = 256)
        self.batchnorm_fc2 = nn.BatchNorm2d(num_features = 256)
        # fc 256 k*k (no batchnorm, no relu)
        self.fc3 = nn.Linear(in_features = 256, out_features = k*k)
        #weight and bias
        self.weight = nn.Parameter(data= torch.zeros(256,k*k), requires_grad=True)
        self.bias = nn.Parameter(data=torch.zeros(1,k*k), requires_grad=True)
        # add bias
        # reshape
    
    def forward(self, x):

        x = nn.functional.relu(self.bachnorm1(self.conv1(x)))
        x = nn.functional.relu(self.bachnorm2(self.conv2(x)))
        x = nn.functional.relu(self.bachnorm3(self.conv3(x)))

        x = self.max_pool(X)

        x = nn.functional.relu(self.bachnorm_fc1(self.fc1(x)))
        x = nn.functional.relu(self.bachnorm_fc2(self.fc2(x)))
        x = self.fc3(x)

        x = x * self.weight
        x = x + self.bias

        x = torch.reshape(x,(self.k,self.k)

        
        return x


class PointNetfeat(nn.Module):
    def __init__(self, global_feat = True, feature_transform = False):
        super(PointNetfeat, self).__init__()
        # Use TNet to apply transformation on input and multiply the input points with the transformation
        # conv 3 64
        # Use TNet to apply transformation on features and multiply the input features with the transformation 
        #                                                                        (if feature_transform is true)
        # conv 64 128
        # conv 128 1024 (no relu)
        # max pool

    def forward(self, x):
        n_pts = x.size()[2]

        # You will need these extra outputs:
        # trans = output of applying TNet function to input
        # trans_feat = output of applying TNet function to features (if feature_transform is true)

        if self.global_feat: # This shows if we're doing classification or segmentation
            return x, trans, trans_feat
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, n_pts)
            return torch.cat([x, pointfeat], 1), trans, trans_feat

class PointNetCls(nn.Module):
    def __init__(self, k = 2, feature_transform=False):
        super(PointNetCls, self).__init__()
        self.feature_transform = feature_transform
        self.feat = PointNetfeat(global_feat=True, feature_transform=feature_transform)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)
        self.dropout = nn.Dropout(p=0.3)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

    def forward(self, x):
        x, trans, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1), trans, trans_feat


class PointNetDenseCls(nn.Module):
    def __init__(self, k = 2, feature_transform=False):
        super(PointNetDenseCls, self).__init__()
        # get global features + point features from PointNetfeat
        # conv 1088 512
        # conv 512 256
        # conv 256 128
        # conv 128 k
        # softmax 
    
    def forward(self, x):
        # You will need these extra outputs: 
        # trans = output of applying TNet function to input
        # trans_feat = output of applying TNet function to features (if feature_transform is true)
        # (you can directly get them from PointNetfeat)
        return x, trans, trans_feat

def feature_transform_regularizer(trans):
    # compute |((trans * trans.transpose) - I)|^2
    
    return loss

if __name__ == '__main__':
    sim_data = Variable(torch.rand(32,3,2500))
    trans = TNet(k=3)
    out = trans(sim_data)
    print('TNet', out.size())
    print('loss', feature_transform_regularizer(out))

    sim_data_64d = Variable(torch.rand(32, 64, 2500))
    trans = TNet(k=64)
    out = trans(sim_data_64d)
    print('TNet 64d', out.size())
    print('loss', feature_transform_regularizer(out))

    pointfeat = PointNetfeat(global_feat=True)
    out, _, _ = pointfeat(sim_data)
    print('global feat', out.size())

    pointfeat = PointNetfeat(global_feat=False)
    out, _, _ = pointfeat(sim_data)
    print('point feat', out.size())

    cls = PointNetCls(k = 5)
    out, _, _ = cls(sim_data)
    print('class', out.size())

    seg = PointNetDenseCls(k = 3)
    out, _, _ = seg(sim_data)
    print('seg', out.size())