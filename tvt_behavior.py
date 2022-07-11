import torch
from torch import nn
from torch.nn import functional as F

from configs import *

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        X = X.to(device) # ------------------------ 批级别训练中传输数据到GPU的一种方式
        y = y.to(device) # --------------------｜
        pred = model(X) #前传播
        loss = loss_fn(pred, y) #误差计算

        # Backpropagation
        optimizer.zero_grad() # torch会默认累计连续batch的梯度。反传播前清除之
        loss.backward() # 反传播
        optimizer.step() # 优化器步进

        if batch % 100 == 0:# 打印
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():# no_grad关闭梯度计算。inference 速度
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)
            pred = model(X) #前传播
            test_loss += loss_fn(pred, y).item() #累计误差
            correct += (pred.argmax(1) == y).type(torch.float).sum().item() #正例数

    test_loss /= num_batches #平均误差
    correct /= size #平均正例数
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")