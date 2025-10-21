# LC25000 Cancer Classification Project
# Author: Amir Aynede
# File: gradcam.py
# Description: Generate Grad-CAM heatmap for a given image

import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from model import build_model

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, class_idx):
        self.model.zero_grad()
        output = self.model(input_tensor)
        target = output[0, class_idx]
        target.backward()

        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1).squeeze()
        cam = torch.relu(cam)
        cam = cam - cam.min()
        cam = cam / cam.max()
        return cam.cpu().numpy()

def generate_gradcam(image_path, model_path="saved_models/best_model.pth", output_path="outputs/gradcam_output.jpg"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(num_classes=5)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    from PIL import Image
    image = Image.open(image_path).convert("RGB")
    input_tensor = preprocess(image).unsqueeze(0).to(device)

    target_layer = list(model.layer4.children())[-1]
    gradcam = GradCAM(model, target_layer)

    with torch.no_grad():
        output = model(input_tensor)
    class_idx = torch.argmax(output, dim=1).item()

    cam = gradcam.generate(input_tensor, class_idx)
    cam_resized = cv2.resize(cam, (224, 224))
    cam_color = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    image_cv = np.array(image)
    overlay = np.float32(cam_color) / 255 + np.float32(image_cv) / 255
    overlay = overlay / np.max(overlay)

    plt.imshow(overlay)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()
    print(f"Grad-CAM saved to {output_path}")

if __name__ == "__main__":
    sample_image = "sample_images/lungaca530.png"  # example
    generate_gradcam(sample_image)
