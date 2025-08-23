import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import os
# import matplotlib.pyplot as plt
import torch.nn.functional as F


# --- định nghĩa lại class VGG và hàm build layers ---
class VGG(nn.Module):
    def __init__(self, features, output_dim):
        super().__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d(7)
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, output_dim),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        h = x.view(x.shape[0], -1)
        x = self.classifier(h)
        return x, h

def get_vgg_layers(config, batch_norm=True):
    layers = []
    in_channels = 3
    for c in config:
        if c == 'M':
            layers += [nn.MaxPool2d(kernel_size=2)]
        else:
            conv2d = nn.Conv2d(in_channels, c, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(c), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = c
    return nn.Sequential(*layers)

# VGG configurations
vgg11_config = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
vgg13_config = [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
vgg16_config = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
vgg19_config = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIM = 2

# # Khởi tạo lại kiến trúc
# layers = get_vgg_layers(vgg11_config, batch_norm=True)
# model = VGG(layers, OUTPUT_DIM).to(device)

# # Load weight đã train
# model.load_state_dict(torch.load("VGG11-model.pt", map_location=device))
# model.eval()
# print("✅ Model loaded successfully!")

# Transform giống khi train
pretrained_size = (224, 224)
pretrained_means = [0.485, 0.456, 0.406]
pretrained_stds = [0.229, 0.224, 0.225]

transform = transforms.Compose([
    transforms.Resize(pretrained_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=pretrained_means, std=pretrained_stds)
])

# Class mapping (ImageFolder sắp xếp theo alphabet -> ['cats','dogs'])
idx_to_class = {0: "Cat", 1: "Dog"}

def predict_image(image_path, model):
    img = Image.open(image_path).convert("RGB")
    img_t = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs, _ = model(img_t)
        probs = F.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, 1)

    return idx_to_class[pred.item()], conf.item(), img

# test_image_path = "cat.3.jpg"

# label, confidence, img = predict_image(test_image_path, model)

# # Hiển thị ảnh + kết quả
# plt.imshow(img)
# plt.axis("off")
# plt.title(f"Prediction: {label} ({confidence*100:.2f}%)")
# plt.show()