# -*- coding: utf-8 -*-

from __future__ import print_function
import torchvision
import torchvision.transforms as transforms
import os
import torch
import time
import sys
from vt_resnet_34 import*
import argparse, pickle
import torch.distributed as dist
from torchvision import models
'''
from apex.parallel import DistributedDataParallel as DDP
from apex.parallel import convert_syncbn_model,SyncBatchNorm
from apex.fp16_utils import *
from apex import amp
from apex.multi_tensor_apply import multi_tensor_applier
'''
os.environ['CUDA_VISIBLE_DEVICES'] = "4"
transform_train = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

])

transform_test = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
def val(optimizer,ann):

    correct = 0
    total = 0

    ann.eval()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
          if ((batch_idx+1)<=len(test_set)//batch_size):
            inputs = inputs.to(device)
            optimizer.zero_grad()

            outputs = ann(inputs)
            loss = criterion(outputs.cpu(), targets)


            _, predicted = outputs.cpu().max(1)
            total += float(targets.size(0))
            correct += float(predicted.eq(targets).sum().item())

    
    acc = 100. * float(correct) / float(total)
    acc_record.append(acc)
   
    if True:#dist.get_rank() == 0:
        state = {
            'net': ann.state_dict(),

            'acc': acc_record,
            'epoch': epoch+start_epoch+1,
            'acc_record': acc_record,
            'optimizer':optimizer.state_dict()
        }
        
        best_acc = acc
        with open('/home/zhl/ZHL/VT+TransT/log_ %d.txt' % (start_epoch+epoch+1),'a') as f:
          f.write('Test Accuracy of the model on the 10000 test images: %.3f \n' % (100 * correct / total))
        torch.save(state, '/home/zhl/ZHL/checkpoint_zhl/ckpt.t7')
    return optimizer
'''
def parse():
  parser = argparse.ArgumentParser()
  parser.add_argument('--local_rank',type = int,default=0)
  args = parser.parse_args()
  return args
args = parse()
'''
batch_size = 160
#torch.cuda.set_device(args.local_rank)
#torch.distributed.init_process_group('nccl',init_method='env://')
def create_model(n_classes):
    model = models.resnet50(pretrained=True)
    model.layer4 = VT(in_channels = 1024, vt_layers_num = 3, tokens = 16,token_channels = 1024,vt_channels = 1024,transformer_enc_layers = 1,transformer_heads = 2 )
    n_features = model.fc.in_features

    model.fc = nn.Linear(n_features // 2, n_classes)
    return model.to(device)
def lr_scheduler(optimizer, epoch, init_lr=0.1, lr_decay_epoch=50):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    if epoch % lr_decay_epoch == 0 and epoch > 1:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.1
    return optimizer
data_path =  '/data/lyh/imagenet2012_png/train/' #todo: input your data path
'''
print(dist.get_world_size())
print(dist.get_rank())
'''
train_dataset = torchvision.datasets.ImageFolder(root= data_path, transform=transform_train)
'''
train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset,num_replicas=dist.get_world_size(),rank = dist.get_rank())
'''
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4,pin_memory=True,drop_last=True)#sampler=train_sampler)
data_path =  '/data/lyh/imagenet2012_png/val/' 
test_set = torchvision.datasets.ImageFolder(root= data_path,transform=transform_test)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4,pin_memory=True)

best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
acc_record = list([])

loss_train_record = list([])
loss_test_record = list([])


ann = create_model(1000)

if True:

    model_CKPT = torch.load('/home/zhl/ZHL/checkpoint_zhl/ckpt.t7',map_location='cpu')
    ann.load_state_dict(model_CKPT['net'])
    print('loading checkpoint!')
    #optimizer.load_state_dict(model_CKPT['optimizer'])
    acc_record = model_CKPT['acc_record']
    start_epoch = model_CKPT['epoch']

#ann=convert_syncbn_model(ann)
ann.to(device)
print(acc_record,start_epoch)
learning_rate = 0.0001
num_epochs = 100
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(ann.parameters(),0.0001,momentum = 0.9)
'''
def reduce_tensor(tensor, world_size=1):
   dist.all_reduce(tensor,op = dist.ReduceOp.SUM)
   tensor /=world_size
   return tensor
'''
#ann,optimizer = amp.initialize(ann , optimizer ,opt_level='O1')
#ann = DDP(ann)
for epoch in range(num_epochs):
    '''
    train_sampler.set_epoch(epoch)
    '''
    running_loss = 0
    start_time = time.time()
    ann.train()  
    for i, (images, labels) in enumerate(train_loader):
        if ((i+1)<=len(train_dataset)//batch_size):
          
          ann.zero_grad()
          optimizer.zero_grad()
          images = images.to(device)
          outputs = ann(images)
         
          loss = criterion(outputs.cpu(), labels)
          running_loss += loss.item()
          #with amp.scale_loss(loss,optimizer) as loss:
          loss.backward()
          optimizer.step()
          if (i+1)%100 == 0:
             with open('/home/zhl/ZHL/VT+TransT/log_ %d.txt' % (start_epoch+epoch+1),'a') as f:
               
               f.write('Epoch [%d/%d], Step [%d/%d], Loss: %.5f \n'
                       %(epoch+1+start_epoch, num_epochs, i+1, len(train_dataset)//batch_size,running_loss ))
               running_loss = 0
               f.write('Time elasped: %d \n'  %(time.time()-start_time))
          torch.cuda.empty_cache()
    val(optimizer,ann)
    optimizer = lr_scheduler(optimizer, epoch+start_epoch+1, learning_rate, 20)
    



        
