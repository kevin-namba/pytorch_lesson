import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
import torchvision
import csv
def fix_seed(seed):
    np.random.seed(int(seed))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
fix_seed(stid)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

transform = transforms.Compose([
  transforms.RandomVerticalFlip(50),
  transforms.RandomHorizontalFlip(50),
  transforms.Grayscale(num_output_channels=1),
  transforms.ToTensor(),
  transforms.Normalize((0.5), (0.5))
  ])
data = torchvision.datasets.ImageFolder(root='./imgdata', transform=transform)
print(len(data))

batch_size = 100
train_size = int(0.8 * len(data))
validation_size  = len(data) - train_size
data_train, data_validation = torch.utils.data.random_split(data,  [560,2240] )
train_loader = DataLoader(data_train, batch_size=batch_size, shuffle=True)
validation_loader = DataLoader(data_validation, batch_size=batch_size, shuffle=False)
label = []
with open('imgdata/labels.csv') as f:
  label = next(csv.reader(f))


def imshow(img):
  # unnormalize [-1, 1] => [0, 1]
  img = img / 2 + 0.5
  img = img.numpy()
  # [c, h, w] => [h, w, c]
  plt.imshow(np.transpose(img, (1, 2, 0)))
images, labels = iter(validation_loader).next()
images, labels = images[:16], labels[:16]
print(images.size())
imshow(torchvision.utils.make_grid(images, nrow=4, padding=1))
plt.axis('off')

class CNN(nn.Module):
  def __init__(self):
    super(CNN, self).__init__()
    self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
    self.pool = nn.MaxPool2d(2, 2)
    self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
    self.fc1 = nn.Linear(16*5*5, 256)
    self.fc2 = nn.Linear(256, 7)
  def forward(self, x):
#    print('0:', x.shape)
    x = self.pool(F.relu(self.conv1(x)))
#    print('1:', x.shape)
    x = self.pool(F.relu(self.conv2(x)))
#    print('2:', x.shape)
    x = x.view(-1, 16*5*5)
#    print('3:', x.shape)
    x = F.relu(self.fc1(x))
#    print('4:', x.shape)
    x = self.fc2(x)
#    print('5:', x.shape)
    return x
def init_weights(m):
  if type(m) == nn.Linear:
    torch.nn.init.xavier_uniform(m.weight)
    m.bias.data.fill_(0.01)
  if type(m) == nn.Conv2d:
    torch.nn.init.xavier_uniform(m.weight)
model = CNN().to(device)

num_epochs = 10
learning_rate = 0.001
criterion = nn.CrossEntropyLoss() 
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
model.apply(init_weights)
def train(train_loader):
  model.train()
  running_loss = 0
  for i, (images, labels) in enumerate(train_loader):
    images, labels = images.to(device), labels.to(device)
    optimizer.zero_grad()
    outputs = model(images)
    loss = criterion(outputs, labels)
    running_loss += loss.item()
    loss.backward()
    optimizer.step()
  train_loss = running_loss / len(train_loader)
  return train_loss
def valid(valication_loader):
  model.eval()
  running_loss = 0
  correct = 0
  total = 0
  with torch.no_grad():
    for i, (images, labels) in enumerate(validation_loader):
      images, labels = images.to(device), labels.to(device)
      outputs = model(images)
      loss = criterion(outputs, labels)
      running_loss += loss.item()
      predicted = outputs.max(1, keepdim=True)[1]
      correct += predicted.eq(labels.view_as(predicted)).sum().item()
      total += labels.size(0)
  val_loss = running_loss / len(validation_loader)
  val_acc = correct / total
  return val_loss, val_acc
loss_list = []
val_loss_list = []
val_acc_list = []
for epoch in range(num_epochs):
  loss = train(train_loader)
  val_loss, val_acc = valid(validation_loader)
  print('epoch %d, loss: %.4f val_loss: %.4f val_acc: %.4f'
      % (epoch, loss, val_loss, val_acc))
  loss_list.append(loss)
  val_loss_list.append(val_loss)
  val_acc_list.append(val_acc)
print('訓練終了')

plt.figure()
plt.plot(range(num_epochs), loss_list, 'r-', label='train_loss')
plt.plot(range(num_epochs), val_loss_list, 'b-', label='test_loss')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.grid()
plt.figure()
plt.plot(range(num_epochs), val_acc_list, 'g-', label='val_acc')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('acc')
plt.grid()
