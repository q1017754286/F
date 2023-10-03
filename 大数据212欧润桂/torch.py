import torchvision
from torch.nn import nn
import torch
from torch.nn import Conv2d,MaxPool2d,Flatten
from torch.utils.data import DataLoader

train_data = torchvision.datasets.CIFAR10(root = "data", train=True,transform=torchvision.transforms.ToTensor(),download=True)
test_data = torchvision.datasets.CIFAR10(root='data',train = False , transform=torchvision.transforms.ToTensor(),)

train_loader = DataLoader(dataset = train_data,batch_size = 64,shuffle = False)
test_loader = DataLoader(dataset = test_data,batch_size = 64)

class Tudui(nn.Module):
    def __init__(self,*arg,**kwargs):
        super().__init__(*arg,**kwargs)
    
    def forward(self,x):
        output = self.model(x)
        return output
    
tudui = Tudui()
cross = nn.CrossEntropyLoss()
optim = torch.optim.SGD(params=tudui.paramters(), lr=0.02) 
total_loss = 0

for epoch in range(20):  
    for data in train_loader:
        img,label = data
        output = tudui(img)
        loss_fn = cross(output,label)

        optim.zero_grad()   #优化器清零
        loss_fn.backward()  #损失函数反向传播   
        optim.step()        #参数更新

def test(loader,model):
    correct = 0 
    num = 0 
    
    for data in loader:
        img,label = data
        output = tudui(img)
        correct += (output.argmax(1) == label).sum()
        num += output.size()
    return correct/num

accuracy = test(test_loader,tudui)

print("准确率为：",accuracy)






