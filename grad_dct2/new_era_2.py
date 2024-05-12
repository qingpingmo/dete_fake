import numpy as np
from scipy.fftpack import dct
from PIL import Image
from scipy.signal import convolve2d
from PIL import Image
import torch
import torch.nn as nn
import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import Compose, Resize, Normalize, ToTensor

def apply_dct(image):
    """Apply DCT (Discrete Cosine Transform) to the given image.

    Args:
        image (PIL.Image): The input image.

    Returns:
        PIL.Image: The DCT transformed image.
    """
    image_np = np.array(image)
    dct_transformed = dct(dct(image_np, axis=0, norm='ortho'), axis=1, norm='ortho')
    dct_image = np.log(np.abs(dct_transformed) + 1)  # Using log scale for better visualization
    dct_image_normalized = (dct_image - np.min(dct_image)) / (np.max(dct_image) - np.min(dct_image))
    image_uint8 = (dct_image_normalized * 255).astype(np.uint8)
    return Image.fromarray(image_uint8)


def generate_gradient_image(image):
    """Generate a gradient image using Sobel operators.

    Args:
        image (PIL.Image): The input image.

    Returns:
        PIL.Image: The gradient image.
    """
    image_np = np.array(image)
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])  # Sobel operator for x-axis
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])  # Sobel operator for y-axis

    grad = np.zeros_like(image_np)
    for i in range(3):  # Apply on each color channel
        grad_x = np.abs(convolve2d(image_np[:, :, i], sobel_x, mode='same', boundary='wrap'))
        grad_y = np.abs(convolve2d(image_np[:, :, i], sobel_y, mode='same', boundary='wrap'))
        grad[:, :, i] = np.sqrt(grad_x ** 2 + grad_y ** 2)

    # Take the maximum over the color channels
    grad = np.max(grad, axis=2)
    gradient_normalized = (grad - np.min(grad)) / (np.max(grad) - np.min(grad))
    gradient_uint8 = (gradient_normalized * 255).astype(np.uint8)

    return Image.fromarray(gradient_uint8)




class DCTFeatureExtractor(nn.Module):
    """Feature extractor for DCT transformed images."""
    def __init__(self):
        super(DCTFeatureExtractor, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        return x

class GradientFeatureExtractor(nn.Module):
    """Feature extractor for gradient images."""
    def __init__(self):
        super(GradientFeatureExtractor, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=2),  # Expecting 3 input channels
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        #print(f"GradientFeatureExtractor Input Shape: {x.shape}")  # Print input shape
        # Expand single channel to 3 to match the expected input shape for convolutional layers
        x = x.repeat(1, 3, 1, 1)  # Repeat the channel dimension
        #print(f"GradientFeatureExtractor Adjusted Input Shape: {x.shape}")  # Print adjusted input shape
        x = self.conv_layers(x)
        #print(f"GradientFeatureExtractor Output Shape: {x.shape}")  # Print output shape
        x = x.view(x.size(0), -1)
        return x
    


class ProGANClassifier(nn.Module):
    """Classifier for ProGAN images, integrating DCT and Gradient features."""
    def __init__(self):
        super(ProGANClassifier, self).__init__()
        # Feature extractors
        self.dct_extractor = DCTFeatureExtractor()
        self.gradient_extractor = GradientFeatureExtractor()
        
        # Feature fusion
        self.fusion_layer = nn.Sequential(
            nn.Linear(64 * 3 * 3 * 2, 512),  # Adjust input size based on feature map sizes
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        
        # Final classification layer
        self.classifier = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 2),  # Binary classification
            nn.Softmax(dim=1)
        )

    def forward(self, dct_image, grad_image):
        #print(f"DCT Image Shape: {dct_image.shape}")  # Print DCT image shape
        #print(f"Grad Image Shape: {grad_image.shape}")  # Print Grad image shape
        dct_features = self.dct_extractor(dct_image)
        grad_features = self.gradient_extractor(grad_image)
        #print(f"DCT Features Shape: {dct_features.shape}")  # Print DCT features shape
        #print(f"Grad Features Shape: {grad_features.shape}") 
        
        # Concatenate features from both extractors
        combined_features = torch.cat((dct_features, grad_features), dim=1)
        
        # Fusion and classification
        fusion_features = self.fusion_layer(combined_features)
        #print(f"Shape: {fusion_features.shape}") 
        output = self.classifier(fusion_features)
        #print(f"Shape: {output.shape}") 
        return output
    

class ProGAN_Dataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir  # 数据集的根目录
        self.transform = transform  # 预处理转换
        self.image_paths = []  # 存储图像路径的列表
        self.labels = []  # 存储图像标签的列表
        self._load_images(self.root_dir)  # 加载图像

    def _load_images(self, current_dir):
        for item in os.listdir(current_dir):  # 遍历当前目录
            path = os.path.join(current_dir, item)  # 获取文件或目录的完整路径
            if os.path.isdir(path):  # 如果是目录，递归调用加载图像
                self._load_images(path)
            elif path.endswith('.png'):  # 如果是PNG图像文件
                label = 1 if '1_fake' in path else 0  # 根据文件名判断图像是否为假图像，并设置标签
                self.image_paths.append(path)  # 添加图像路径到列表
                self.labels.append(label)  # 添加标签到列表

    def __len__(self):
        return len(self.image_paths)  # 返回数据集中图像的数量

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]  # 根据索引获取图像路径
        image = Image.open(img_path).convert('RGB')  # 打开图像并转换为RGB格式
        dct_image = apply_dct(image)  # 对图像应用DCT变换
        grad_image = generate_gradient_image(image)  # 生成图像的梯度图

        if self.transform:  # 如果设置了预处理转换
            dct_image = self.transform(dct_image)  # 对DCT图像应用预处理转换
            grad_image = self.transform(grad_image)  # 对梯度图应用预处理转换

        label = self.labels[idx]  # 获取图像标签
        return dct_image, grad_image, label  # 返回DCT图像、梯度图和标签




def train_model(model, data_loader, criterion, optimizer, num_epochs=25):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        
        running_loss = 0.0
        for dct_images, grad_images, labels in data_loader:
            dct_images = dct_images.to(device)
            grad_images = grad_images.to(device)
            labels = labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward
            outputs = model(dct_images, grad_images)
            loss = criterion(outputs, labels)

            # Backward + optimize
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(data_loader.dataset)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')

    print('Training complete')



def evaluate_model(model, data_loader):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()  # Set model to evaluate mode

    correct = 0
    total = 0
    with torch.no_grad():
        for dct_images, grad_images, labels in data_loader:
            dct_images = dct_images.to(device)
            grad_images = grad_images.to(device)
            labels = labels.to(device)

            outputs = model(dct_images, grad_images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy of the model on the test images: {accuracy:.2f}%')



transform = Compose([
    Resize((256, 256)),
    ToTensor(),
    Normalize(mean=[0.5], std=[0.5]) 
])


def main():
    
    # Initialize the data loader
    train_dataset = ProGAN_Dataset(root_dir='/opt/data/private/wangjuntong/datasets/progan_train', transform=transform)
    # 创建训练数据加载器
    trainloader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=5, pin_memory=True)

    # Initialize the model
    model = ProGANClassifier()

    # Define the loss function
    criterion = nn.CrossEntropyLoss()

    # Define the optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Number of epochs to train the model
    num_epochs = 10

    # Train the model
    train_model(model, trainloader, criterion, optimizer, num_epochs=num_epochs)


    folders = ['progan', 'stylegan', 'stylegan2', 'biggan', 'cyclegan', 'stargan', 'gaugan', 'deepfake']
        # 遍历每个文件夹进行验证
    for folder in folders:
        print(f"Validating on {folder}...")


            # Initialize the data loader for test data
        val_dataset_folder = os.path.join('/opt/data/private/wangjuntong/datasets/CNN_synth_testset', folder)
                    # 创建验证数据集实例
        val_dataset = ProGAN_Dataset(root_dir=val_dataset_folder, transform=transform)
                    # 创建验证数据加载器
        valloader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=5, pin_memory=True)

            # Initialize the model
        model = ProGANClassifier()

            # Load the trained model weights
        model.load_state_dict(torch.load('path/to/model_weights.pth'))

            # Evaluate the model
        evaluate_model(model, valloader)

if __name__ == '__main__':
    main()

