import os
import sys
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.io import read_image
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import torch
from torch import nn
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models import resnet18
from torch.utils.data import DataLoader
from lib.model.faster_rcnn.faster_rcnn import _fasterRCNN
import torch.nn.functional as F

if torch.cuda.is_available():
    device = torch.device("cuda:3")  
    print("Using GPU:", torch.cuda.get_device_name(3))
else:
    device = torch.device("cpu")
    print("CUDA is not available. Using CPU.")



class CustomFasterRCNN(nn.Module):
    def __init__(self, classes, class_agnostic):
        super().__init__()
        self.faster_rcnn = _fasterRCNN(classes, class_agnostic)
    
    def forward(self, im_data):
        
        base_feat = self.faster_rcnn.RCNN_base3(im_data)
        return base_feat




class GradientReversalLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -grad_output


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.feature_extractor = fasterrcnn_resnet50_fpn(pretrained=True).backbone

    def forward(self, x):
        x = self.feature_extractor(x)
        return x


class Classifier1(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(1000, 2)

    def forward(self, x):
        x = self.fc(x)
        return x


class Classifier2(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.fc = nn.Linear(1000, 2)
        self.grl = GradientReversalLayer.apply

    def forward(self, x):
        x = self.grl(x)
        x = self.fc(x)
        return x


class MyModel(nn.Module):
    def __init__(self, classes, class_agnostic):
        super().__init__()
        self.encoder = CustomFasterRCNN(classes, class_agnostic)
        self.classifier1 = Classifier1()
        self.classifier2 = Classifier2()

    def forward(self, im_data):
        features = self.encoder(im_data) 
        class1_pred = self.classifier1(features)
        class2_pred = self.classifier2(features)
        return class1_pred, class2_pred




class ImageDataset(Dataset):
    def __init__(self, real_dir, biggan_dir, which_dir):
        self.real_images = [os.path.join(real_dir, file) for file in os.listdir(real_dir)]
        self.biggan_images = [os.path.join(biggan_dir, file) for file in os.listdir(biggan_dir)]
        self.which_images = [os.path.join(which_dir, file) for file in os.listdir(which_dir)]
        self.total_images = self.real_images + self.biggan_images + self.which_images
        self.labels = [0]*len(self.real_images) + [1]*len(self.biggan_images) + [2]*len(self.which_images)

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.total_images)

    def __getitem__(self, idx):
        image_path = self.total_images[idx]
        
        image = Image.open(image_path).convert('RGB')

        image = self.transform(image)

        label = self.labels[idx]
        label1 = 0 if label == 0 else 1 
        label2 = label - 1 if label > 0 else -1  

        return image, (label1, label2)

def loss_fn(class1_pred, class2_pred, labels1, labels2):
    loss1 = F.cross_entropy(class1_pred, labels1)
    valid_indices = labels2 != -1
    loss2 = F.cross_entropy(class2_pred[valid_indices], labels2[valid_indices]) if valid_indices.any() else 0
    total_loss = loss1 + loss2
    return total_loss


classes = ['real', 'fake_gan1', 'fake_gan2']  
class_agnostic = False


epochs = 10  

real_dir = 'real'
biggan_dir = 'biggan'
which_dir = 'which'
dataset = ImageDataset(real_dir, biggan_dir, which_dir)

data_loader = DataLoader(dataset, batch_size=4, shuffle=True)


model = MyModel(classes, class_agnostic)
model.to(device) 
optimizer = torch.optim.Adam(model.parameters())


def calculate_accuracy(predictions, labels):
    _, predicted = torch.max(predictions, 1)
    correct = (predicted == labels).sum().item()
    return correct / labels.size(0)

def train(model, data_loader, loss_fn, optimizer, epochs, device):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        total_correct1 = 0
        total_correct2 = 0
        total_samples1 = 0
        total_samples2 = 0

        for batch_idx, (images, (labels1, labels2)) in enumerate(data_loader):
            images = images.to(device)
            labels1 = labels1.to(device)
            labels2 = labels2.to(device)

            
            class1_pred, class2_pred = model(images)

            
            loss = loss_fn(class1_pred, class2_pred, labels1, labels2)
            total_loss += loss.item()

            
            total_correct1 += (class1_pred.argmax(1) == labels1).sum().item()
            total_samples1 += labels1.size(0)

            
            valid_indices = labels2 != -1
            if valid_indices.any():
                total_correct2 += (class2_pred[valid_indices].argmax(1) == labels2[valid_indices]).sum().item()
                total_samples2 += valid_indices.sum().item()

            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Batch [{batch_idx+1}/{len(data_loader)}], Loss: {loss.item()}")

        avg_loss = total_loss / len(data_loader)
        avg_accuracy1 = total_correct1 / total_samples1 * 100
        avg_accuracy2 = total_correct2 / total_samples2 * 100 if total_samples2 > 0 else 0
        print(f"End of Epoch [{epoch+1}/{epochs}], Average Loss: {avg_loss}, Accuracy for Classifier1: {avg_accuracy1:.2f}%, Accuracy for Classifier2: {avg_accuracy2:.2f}%")


train(model, data_loader, loss_fn, optimizer, epochs, device)


