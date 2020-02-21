import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

train_s = torchvision.datasets.FashionMNIST(root = './data/FashionMNIST',train=True,download=True,transform = transforms.Compose([transforms.ToTensor()]))
trainloader = torch.utils.data.DataLoader(train_s, batch_size=4,shuffle=True)
class classify3(nn.Module):
  def __init__(self):
    super(classify3,self).__init__()
    self.conv1 = nn.Conv2d(1,32,3)
    self.mx = nn.MaxPool2d(2,2)
    self.fc1 = nn.Linear(5408,100)
    self.fc2 = nn.Linear(100,10)

  def forward(self,x):
    x = (self.mx(F.relu(self.conv1(x))))
    x = torch.flatten(x,start_dim = 1)
    return self.fc2(F.relu(self.fc1(x)))

class classify(nn.Module):
  def __init__(self):
    super(classify,self).__init__()
    self.fc1 = nn.Linear(784,200)
    self.fc2 = nn.Linear(200,100)
    self.fc3 = nn.Linear(100,10)
  #Vishnu
  def forward(self,x):
    x = self.fc3(F.relu(self.fc2(F.relu(self.fc1(x)))))
    return x

net = classify()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
net.to(device)
net2 = classify3()
net2.to(device)

criterion = nn.CrossEntropyLoss()
criterion1 = nn.CrossEntropyLoss()
optimizer1 = optim.SGD(net.parameters(),lr = 0.001, momentum=0.9)
optimizer = optim.SGD(net2.parameters(),lr = 0.001,momentum=0.9)

# trial2
path = 'model'
for epoch in range(35):
  r_l1 = 0.0
  r_l = 0.0
  torch.save(net.state_dict(), path+"/net_nn_"+str(epoch)+".pth")
  torch.save(net2.state_dict(), path+"/net_cnn_"+str(epoch)+".pth")

  for i,data in enumerate(trainloader,0):
    inputs,labels_t = data
    inputs1 = torch.flatten(inputs,start_dim=1)
    inputs1,labels_t = inputs1.to(device), labels_t.to(device)
    inputs = inputs.to(device)

    optimizer.zero_grad()
    optimizer1.zero_grad()

    outputs1 = net(inputs1)
    outputs = net2(inputs)

    loss1 = criterion1(outputs1,labels_t)
    loss = criterion(outputs,labels_t)

    loss1.backward()
    loss.backward()

    optimizer1.step()
    optimizer.step()

    r_l1 += loss.item()
    r_l += loss.item()

  correct1 = 0
  correct = 0
  total1 = 0
  total = 0

  for data in testloader:
    input_s,label_s = data
    input_s1 = torch.flatten(input_s,start_dim=1)
    input_s1,label_s = input_s1.to(device), label_s.to(device)
    input_s = input_s.to(device)

    output_s1 = net(input_s1)
    output_s = net2(input_s)

    _,predicted1 = torch.max(output_s1.data,1)
    _,predicted = torch.max(output_s.data,1)

    total += label_s.size(0)
    total1 += label_s.size(0)

    correct1 += (predicted1 == label_s).sum().item()
    correct += (predicted == label_s).sum().item()

  print('[%d nn] loss: %.3f test_accuracy: %.2f' % (epoch+1, r_l1 / (i+1), (100*correct1)/total1))
  print('[%d cnn] loss: %.3f test_accuracy: %.2f'% (epoch+1, r_l/(i+1), (100*correct)/total))
  r_l = 0.0
  r_l1 = 0.0


print('trained')
