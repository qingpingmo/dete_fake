import torch
import torchvision.transforms as transforms
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import numpy as np
from scipy.fftpack import dct, idct
import torch.nn as nn
from math import sqrt
from scipy.signal import convolve2d
from torchvision.models import resnet
import torch.nn.functional as F



os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def apply_dct(image):
    image_np = np.array(image)
    dct_transformed = dct(dct(image_np, axis=0, norm='ortho'), axis=1, norm='ortho')
    # 将 DCT 变换的结果转换为图像格式，保持大小不变
    dct_image = np.log(np.abs(dct_transformed) + 1)  # 使用对数尺度以便更好地可视化
    dct_image_normalized = (dct_image - np.min(dct_image)) / (np.max(dct_image) - np.min(dct_image))
    image_uint8 = (dct_image_normalized * 255).astype(np.uint8)
    return Image.fromarray(image_uint8)

def generate_gradient_image(image):
    image_np = np.array(image)
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    grad = np.zeros_like(image_np)
    for i in range(3):  # 对每个颜色通道分别应用
        grad_x = np.abs(convolve2d(image_np[:, :, i], sobel_x, mode='same', boundary='wrap'))
        grad_y = np.abs(convolve2d(image_np[:, :, i], sobel_y, mode='same', boundary='wrap'))
        grad[:, :, i] = np.sqrt(grad_x ** 2 + grad_y ** 2)

    grad = np.max(grad, axis=2)
    gradient_normalized = (grad - np.min(grad)) / (np.max(grad) - np.min(grad))
    gradient_uint8 = (gradient_normalized * 255).astype(np.uint8)

    # 复制单通道图像到3个通道
    gradient_3channel = np.stack([gradient_uint8]*3, axis=-1)

    return Image.fromarray(gradient_3channel)




class DCTFeatureExtractor(nn.Module):
    def __init__(self):
        super(DCTFeatureExtractor, self).__init__()
        # Define the convolutional layers for DCT feature extraction
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        # Apply the layers defined in the constructor
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.relu(self.conv3(x))
        x = self.pool(x)
        return x

class GradientFeatureExtractor(nn.Module):
    def __init__(self):
        super(GradientFeatureExtractor, self).__init__()
        # Define the convolutional layers for Gradient feature extraction, with a slightly different architecture
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        # Apply the layers defined in the constructor
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.relu(self.conv3(x))
        x = self.pool(x)
        return x

class ProGAN_Dataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self._load_images(self.root_dir)
        # Initialize the DCT and Gradient feature extractors
        self.dct_extractor = DCTFeatureExtractor()
        self.grad_extractor = GradientFeatureExtractor()

    def _load_images(self, current_dir):
        for item in os.listdir(current_dir):
            path = os.path.join(current_dir, item)
            if os.path.isdir(path):
                self._load_images(path)
            elif path.endswith('.png'):
                label = 1 if '1_fake' in path else 0
                self.image_paths.append(path)
                self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        dct_image = apply_dct(image)
        grad_image = generate_gradient_image(image)

        if self.transform:
            dct_image = self.transform(dct_image)
            grad_image = self.transform(grad_image)

        # 增加一个批次维度
        dct_image = dct_image.unsqueeze(0)  # Now shape is [1, channels, height, width]
        grad_image = grad_image.unsqueeze(0)  # Now shape is [1, channels, height, width]

        dct_features = self.dct_extractor(dct_image)
        grad_features = self.grad_extractor(grad_image)

        # 自适应池化来调整 grad_features 尺寸以匹配 dct_features
        grad_features = F.adaptive_avg_pool2d(grad_features, (2, 2))
        #print(f"dct_features shape right after DCTFeatureExtractor: {dct_features.shape}")
        #print(f"grad_features shape right after GradientFeatureExtractor: {grad_features.shape}")
        label = self.labels[idx]
        return dct_features.squeeze(0), grad_features.squeeze(0), label






class JointRepresentationModel(nn.Module):
    def __init__(self):
        super(JointRepresentationModel, self).__init__()
        
        # Assuming these are the correct feature sizes based on your model architecture
        self.grad_feature_size = (64, 2, 2)  # Updated to match actual output from GradientFeatureExtractor
        self.dct_feature_size = (64, 2, 2)   # Matches actual output from DCTFeatureExtractor

        # Calculate the total number of elements in the feature maps
        self.grad_feature_elements = np.prod(self.grad_feature_size)
        self.dct_feature_elements = np.prod(self.dct_feature_size)

        # Feature Fusion: Adjust the input size to match the total number of feature elements
        self.fusion = nn.Linear(self.grad_feature_elements + self.dct_feature_elements, 1024)  # Updated input size

        # Common Representation Learning
        self.common_rep = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )

        # Decoders for reconstructing Gradient and DCT images
        self.decoder_grad = nn.Sequential(
            nn.Linear(256, self.grad_feature_elements),
            nn.ReLU()
        )

        self.decoder_dct = nn.Sequential(
            nn.Linear(256, self.dct_feature_elements),
            nn.ReLU()
        )

        # Classifier: A fully connected layer for classification from the common representation space
        self.classifier = nn.Sequential(
            nn.Linear(256, 1),
            nn.Sigmoid()  # Use Sigmoid for binary classification
        )

    def forward(self, dct_features, grad_features):
        # Flatten features for fusion
        dct_flat = torch.flatten(dct_features, start_dim=1)
        grad_flat = torch.flatten(grad_features, start_dim=1)

        # Concatenate features
        fused = torch.cat((dct_flat, grad_flat), dim=1)

        # Feature Fusion
        fused = F.relu(self.fusion(fused))

        # Common Representation Learning
        common_rep = self.common_rep(fused)

        # Decode to reconstruct Gradient and DCT features
        recon_grad = self.decoder_grad(common_rep)
        recon_dct = self.decoder_dct(common_rep)

        # Classification
        classification = self.classifier(common_rep)

        # Reshape to original feature map sizes for reconstruction
        recon_grad_reshaped = recon_grad.view(-1, *self.grad_feature_size)
        recon_dct_reshaped = recon_dct.view(-1, *self.dct_feature_size)

        return recon_grad_reshaped, recon_dct_reshaped, classification  # Return classification result in addition to reconstructions








    

def train_model(model, train_loader, optimizer, epoch):
    model.train()  # Set model to training mode
    total_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

    for batch_idx, (dct_features, grad_features, labels) in enumerate(train_loader):
        dct_features, grad_features, labels = dct_features.to(device), grad_features.to(device), labels.to(device)
        
        optimizer.zero_grad()  # Zero the parameter gradients
        
        # Forward pass
        recon_grad, recon_dct, classification = model(dct_features, grad_features)

        # Calculate reconstruction losses for grad and dct features
        recon_loss_grad = nn.MSELoss()(recon_grad, grad_features)
        recon_loss_dct = nn.MSELoss()(recon_dct, dct_features)

        # Compute classification loss (Binary Cross Entropy)
        classification_loss = nn.BCELoss()(classification.squeeze(), labels.float())

        # Compute representation loss as the mean cosine similarity loss across the batch
        rep_loss = 1 - nn.CosineSimilarity(dim=1)(torch.flatten(grad_features, start_dim=1), torch.flatten(dct_features, start_dim=1)).mean()

        # Compute total loss
        lambda_rep = 0.1
        lambda_class = 1.0  # Weight for classification loss, adjust as necessary
        loss = recon_loss_grad + recon_loss_dct + lambda_rep * rep_loss + lambda_class * classification_loss

        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Calculate accuracy
        predicted_labels = classification.round()  # Round the classification output to 0 or 1
        correct_predictions += (predicted_labels.squeeze() == labels).sum().item()
        total_predictions += labels.size(0)

        if batch_idx % 10 == 0:  # Print loss every 10 batches
            print(f'Train Epoch: {epoch} [{batch_idx * len(dct_features)}/{len(train_loader.dataset)}'
                  f' ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

    epoch_loss = total_loss / len(train_loader)
    epoch_accuracy = 100. * correct_predictions / total_predictions
    print(f'====> Epoch: {epoch} Average loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%')











def main():
    # Hyperparameters
    batch_size = 128
    epochs = 100
    learning_rate = 1e-3

    

    # Transformations for the images
    transform = transforms.Compose([
        transforms.Resize((64, 64)),  # Resize images to a fixed size for simplicity
        transforms.ToTensor()
    ])

    # Dataset and DataLoader
    train_dataset = ProGAN_Dataset(root_dir='/opt/data/private/wangjuntong/datasets/progan_train', transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Model
    model = JointRepresentationModel().to(device)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    for epoch in range(1, epochs + 1):
        train_model(model, train_loader, optimizer, epoch)

if __name__ == "__main__":
    main()
