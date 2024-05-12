import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import timm



class CLIPModel(nn.Module):
    def __init__(self, temperature, image_embedding):
        super().__init__()
        # Separate image encoders for each image type
        self.real_image_encoder = ImageEncoder()
        self.fake_image_encoder = ImageEncoder()
        self.real_grad_image_encoder = ImageEncoder()
        self.fake_grad_image_encoder = ImageEncoder()
        self.real_dct_image_encoder = ImageEncoder()
        self.fake_dct_image_encoder = ImageEncoder()

        # Projection head to project image features to a common embedding space
        self.projection_head = ProjectionHead(embedding_dim=image_embedding)

        self.temperature = temperature

    def forward(self, real_images, fake_images, real_grad_images, fake_grad_images, real_dct_images, fake_dct_images):
        # Encode images
        real_image_features = self.real_image_encoder(real_images)
        fake_image_features = self.fake_image_encoder(fake_images)
        real_grad_image_features = self.real_grad_image_encoder(real_grad_images)
        fake_grad_image_features = self.fake_grad_image_encoder(fake_grad_images)
        real_dct_image_features = self.real_dct_image_encoder(real_dct_images)
        fake_dct_image_features = self.fake_dct_image_encoder(fake_dct_images)

        # Project image features
        real_image_embeddings = self.projection_head(real_image_features)
        fake_image_embeddings = self.projection_head(fake_image_features)
        real_grad_image_embeddings = self.projection_head(real_grad_image_features)
        fake_grad_image_embeddings = self.projection_head(fake_grad_image_features)
        real_dct_image_embeddings = self.projection_head(real_dct_image_features)
        fake_dct_image_embeddings = self.projection_head(fake_dct_image_features)

        # Calculate loss
        loss = self.calculate_loss(
            real_image_embeddings, fake_image_embeddings, 
            real_grad_image_embeddings, fake_grad_image_embeddings, 
            real_dct_image_embeddings, fake_dct_image_embeddings
        )

        return loss

    def calculate_loss(self, *embeddings):
        # Combine all embeddings and calculate similarity matrix
        combined_embeddings = torch.cat(embeddings, dim=0)
        similarity_matrix = torch.matmul(combined_embeddings, combined_embeddings.T)

        # Scale similarities by temperature
        similarity_matrix /= self.temperature

        # Create target distribution
        batch_size = embeddings[0].shape[0]
        num_embeddings = len(embeddings)
        targets = torch.eye(batch_size * num_embeddings).to(combined_embeddings.device)

        # Calculate contrastive loss
        loss = nn.CrossEntropyLoss()(similarity_matrix, targets)
        return loss

    


class CustomDataset(Dataset):
    def __init__(self, real_dir, fake_dir, real_grad_dir, fake_grad_dir, real_dct_dir, fake_dct_dir, transform=None):
        self.real_images = [os.path.join(real_dir, f) for f in os.listdir(real_dir)]
        self.fake_images = [os.path.join(fake_dir, f) for f in os.listdir(fake_dir)]
        self.real_grad_images = [os.path.join(real_grad_dir, f) for f in os.listdir(real_grad_dir)]
        self.fake_grad_images = [os.path.join(fake_grad_dir, f) for f in os.listdir(fake_grad_dir)]
        self.real_dct_images = [os.path.join(real_dct_dir, f) for f in os.listdir(real_dct_dir)]
        self.fake_dct_images = [os.path.join(fake_dct_dir, f) for f in os.listdir(fake_dct_dir)]
        self.transform = transform

    def __len__(self):
        # Ensure all categories have the same number of images
        return min(len(self.real_images), len(self.fake_images), len(self.real_grad_images), len(self.fake_grad_images), len(self.real_dct_images), len(self.fake_dct_images))

    def __getitem__(self, idx):
        real_image = Image.open(self.real_images[idx]).convert('RGB')
        fake_image = Image.open(self.fake_images[idx]).convert('RGB')
        real_grad_image = Image.open(self.real_grad_images[idx]).convert('RGB')
        fake_grad_image = Image.open(self.fake_grad_images[idx]).convert('RGB')
        real_dct_image = Image.open(self.real_dct_images[idx]).convert('RGB')
        fake_dct_image = Image.open(self.fake_dct_images[idx]).convert('RGB')

        if self.transform:
            real_image = self.transform(real_image)
            fake_image = self.transform(fake_image)
            real_grad_image = self.transform(real_grad_image)
            fake_grad_image = self.transform(fake_grad_image)
            real_dct_image = self.transform(real_dct_image)
            fake_dct_image = self.transform(fake_dct_image)

        return real_image, fake_image, real_grad_image, fake_grad_image, real_dct_image, fake_dct_image

    


class ProjectionHead(nn.Module):
    def __init__(self, embedding_dim=2048, projection_dim=512, dropout=0.1):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)
    
    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x


class ImageEncoder(nn.Module):
    """
    Encode images to a fixed size vector.
    """
    def __init__(self, model_name='resnet50', pretrained=True, trainable=True):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=0, global_pool='avg')
        for p in self.model.parameters():
            p.requires_grad = trainable

    def forward(self, x):
        return self.model(x)
    

# Transformations
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# Dataset and DataLoader
dataset = CustomDataset(
    real_dir="deepfake/0_real", 
    fake_dir="deepfake/1_fake", 
    real_grad_dir="deepfake/0_real_grad", 
    fake_grad_dir="deepfake/1_fake_grad",
    real_dct_dir="deepfake/0_real_dct",
    fake_dct_dir="deepfake/1_fake_dct",
    transform=transform
)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)





# 在这里，确保你实例化了正确的CLIPModel类
model = CLIPModel(temperature=0.07, image_embedding=512)

# 确保模型调用是正确的
epochs = 10
for epoch in range(epochs):
    for real_images, fake_images, real_grad_images, fake_grad_images, real_dct_images, fake_dct_images in dataloader:
        optimizer.zero_grad()

        # 确保这里传递了正确数量的参数
        loss = model(real_images, fake_images, real_grad_images, fake_grad_images, real_dct_images, fake_dct_images)

        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item()}")
