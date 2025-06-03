#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
from sklearn import metrics

from phe import paillier
import tenseal as ts

def make_laplace_noise(size, sensitivity, epsilon):
      
    # 生成拉普拉斯噪聲
    laplace_noise = torch.from_numpy(np.random.laplace(0, sensitivity / epsilon, size))    
    return laplace_noise

def make_staircase_noise(size, sensitivity, epsilon, gamma):
    """
    生成符合 Staircase 機制的雜訊

    參數：
    - size: 雜訊的形狀 (例如 [1000] 表示生成 1000 個雜訊樣本)
    - sensitivity: 查詢的全域敏感度
    - epsilon: 隱私預算
    - gamma: 控制階梯內部均勻分布的比例參數（默認值 1.0）

    返回：
    - 生成的 staircase 雜訊張量
    """
    # 縮放參數
    b = sensitivity / epsilon

    # 生成 ±1 符號
    sign = torch.randint(0, 2, size) * 2 - 1  # 符號 ±1

    # 幾何隨機變數 G
    p = 1 - torch.exp(-torch.tensor(epsilon, dtype=torch.float32))  # 幾何分布參數
    geometric_rv = torch.distributions.Geometric(p).sample(size) - 1  # 幾何分布值

    # 均勻隨機變數 U
    uniform_rv = torch.rand(size)

    # 二元隨機變數 B
    b_prob = gamma / (gamma + (1 - gamma) * torch.exp(torch.tensor(-epsilon)))
    binary_rv = torch.bernoulli(torch.full(size, b_prob))  # 生成二元隨機變數

    # 計算最終噪聲
    noise = sign * ((1 - binary_rv) * (geometric_rv + gamma * uniform_rv) * b +
                    binary_rv * (geometric_rv + gamma + (1 - gamma) * uniform_rv) * b)

    return noise

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label

def encode_gradient(gradient, scale=1e6, encoding_space=2**32, clip_range=(-10.0,10.0)):

    encoded_gradient = {}
    
    for key, value in gradient.items():
        # 確保梯度是 numpy array 格式
        if isinstance(value, torch.Tensor):
            value = value.cpu().numpy()  # 將 PyTorch 張量轉為 NumPy 陣列
        
        # 1. 對梯度進行裁剪
        clipped_gradient = np.clip(value, clip_range[0], clip_range[1])
        
        # 2. 縮放浮點數
        scaled_gradient = (clipped_gradient * scale).astype(int)


        # 3. 編碼正負數
        encoded_gradient[key] = np.where(
            scaled_gradient >= 0,
            scaled_gradient,  # 正數保持不變
            scaled_gradient + encoding_space  # 負數加上編碼空間
        )
    
    return encoded_gradient

def encrypt_gradient(encoded_gradient, public_key):

    encrypted_gradient = {}
    
    for key, value in encoded_gradient.items():
        # 確保數據是 numpy array 格式
        if isinstance(value, torch.Tensor):
            value = value.cpu().numpy()  # 將 PyTorch 張量轉為 NumPy 陣列
        
        # 展平數組並加密每個值
        encrypted_gradient[key] = [public_key.encrypt(int(v)) for v in value.flatten()]
    
    return encrypted_gradient





class LocalUpdate(object):
    def __init__(self, args, dataset=None, idxs=None, public_key=None, private_key=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)
        self.public_key = public_key
        self.private_key = private_key

    def train(self, net):
        
        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)
              
        # Initial gradients    
        net.train()    
        grad = {}
        for name, param in net.named_parameters():
            grad[name] = torch.zeros(param.shape, dtype=torch.float32).to(self.args.device)


        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                optimizer.zero_grad()
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                labels = labels.long()
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                # clip gradients
                #nn.utils.clip_grad_norm(net.parameters(), 3.0)
                optimizer.step()
                for name, param in net.named_parameters():
                    grad[name] += param.grad



                if self.args.verbose and batch_idx % 10 == 0:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter, batch_idx * len(images), len(self.ldr_train.dataset),
                               100. * batch_idx / len(self.ldr_train), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))

 
        
        # add the differential privacy noise to gradient 
        if self.args.dp_mechanism == 'None' :
            print('No DP')
        else :      
            print(self.args.dp_mechanism + ', Epsilon: ' + str(self.args.epsilon ) )          
            noise = {}
            for name in grad.keys():
                if self.args.dp_mechanism == 'laplace':
                    noise[name] = make_laplace_noise(grad[name].size(), sensitivity=1.0, epsilon= self.args.epsilon )
                    grad[name] += noise[name].to(self.args.device)
                elif self.args.dp_mechanism == 'staircase':
                    noise[name] = make_staircase_noise(grad[name].size(), sensitivity=1.0, epsilon= self.args.epsilon, gamma=0.01 )
                    grad[name] += noise[name].to(self.args.device)
                else :
                    exit('Error : There is no dp-mechanism : ' + self.args.dp_mechanism )

                
        ''' 
        only add dp noise in bias instead of weight and bias.    
        noise = {}
        for name in grad.keys():
            if 'bias' in name:
                noise[name] = make_laplace_noise(grad[name].size(), sensitivity=1.0, epsilon=1)
                grad[name] += noise[name].to(self.args.device)
        '''


        
        # Encrypt gradients using Paillier encryption
        print( 'Start to encrypt grad \n')
        encoded_grad = encode_gradient(grad)
        encrypted_grad = encrypt_gradient(encoded_grad, self.public_key)
        print( 'Finishing \n')
        #return net.state_dict(), sum(epoch_loss) / len(epoch_loss)
        return encrypted_grad, sum(epoch_loss) / len(epoch_loss)   

