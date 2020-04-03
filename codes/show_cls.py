from __future__ import print_function
import argparse
import torch
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
from dataset import ShapeNetDataset
from model import PointNetCls
import torch.nn.functional as F

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default = 'clsFeat/cls_model_99.pth',  help='model path')
parser.add_argument('--num_points', type=int, default=2500, help='input batch size')

opt = parser.parse_args()
print(opt)

data_root = '../../../shapenetcore_partanno_segmentation_benchmark_v0'

test_dataset    = ShapeNetDataset(root=data_root,classification=True,npoints=opt.num_points,split='test',data_augmentation=False)
test_dataloader = torch.utils.data.DataLoader(test_dataset,batch_size=32,shuffle=True,num_workers=4)

train_dataset    = ShapeNetDataset(root=data_root,classification=True,npoints=opt.num_points)
train_dataloader = torch.utils.data.DataLoader(train_dataset,batch_size=32,shuffle=True, num_workers=4)

classifier = PointNetCls(k=len(test_dataset.classes),feature_transform=True)
classifier.cuda()
classifier.load_state_dict(torch.load(opt.model))
classifier.eval()

loss = 0
correct = 0

for i, data in enumerate(train_dataloader, 0):
    points, target = data
    target = target[:, 0]
    points = points.transpose(2, 1)
    points, target = points.cuda(), target.cuda()
    pred, trans, trans_feat = classifier(points)
    loss += F.nll_loss(pred, target).item()
    pred_choice = pred.data.max(1)[1]
    correct += pred_choice.eq(target.data).cpu().sum().item()

print('Total train accuracy with feature transform: %f' % (correct /len(train_dataset)))

loss = 0
correct = 0

for i, data in enumerate(test_dataloader, 0):
    points, target = data
    target = target[:, 0]
    points = points.transpose(2, 1)
    points, target = points.cuda(), target.cuda()
    pred, trans, trans_feat = classifier(points)
    loss += F.nll_loss(pred, target).item()
    pred_choice = pred.data.max(1)[1]
    correct += pred_choice.eq(target.data).cpu().sum().item()

print('Total test accuracy with feature transform: %f' % (correct / len(test_dataset)))
