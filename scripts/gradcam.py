"""
When you run your Grad-CAM script, you typically end up with two output files:
	1.	The main Grad-CAM output:
	•	This is the raw image with the heatmap superimposed on top (colored overlay).
	•	Usually saved as something like lungaca492_gradcam.jpg.
	•	This is what you usually want to inspect to see where the model focused.
	2.	The debug heatmap:
	•	This is the standalone heatmap (grayscale or colored by plt.imsave with the JET colormap).
	•	Saved separately as debug_heatmap.png (or similar).
	•	This helps you verify the heatmap itself, independent of the original image, for troubleshooting or visualization.
"""


import torch
import torch.nn.functional as F
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import os
import argparse
from scripts.model import build_model


# Grad-CAM extractor class
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_handles = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            print("✅ Forward hook triggered")
            print(f"output.requires_grad: {output.requires_grad}")
            # Manually enable gradient tracking
            if not output.requires_grad:
                output.requires_grad_(True)
            self.activations = output

        def backward_hook(module, grad_input, grad_output):
            print("✅ Backward hook triggered")
            if grad_output[0] is None:
                print("❌ grad_output[0] is None!")
            else:
                self.gradients = grad_output[0]

        self.hook_handles.append(self.target_layer.register_forward_hook(forward_hook))
        self.hook_handles.append(self.target_layer.register_full_backward_hook(backward_hook))

    def remove_hooks(self):
        for handle in self.hook_handles:
            handle.remove()

    def __call__(self, input_tensor, class_idx=None):
        self.model.zero_grad()
        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        print(f"🎯 Target class index: {class_idx}")
        loss = output[:, class_idx]
        print(f"📉 Loss shape: {loss.shape}")
        loss.backward()

        if self.gradients is None:
            print("❌ Error: gradients are None after backward pass!")
            return None

        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        activations = self.activations[0]

        with torch.no_grad():
            activations = activations * pooled_gradients[:, None, None]

        heatmap = torch.mean(activations, dim=0).cpu().numpy()
        heatmap = np.maximum(heatmap, 0)
    
        # Normalize and apply a gamma correction to boost contrast
        heatmap = heatmap / (np.max(heatmap) + 1e-8)
        heatmap = np.power(heatmap, 0.5)  # Gamma correction (sqrt) - adjust gamma <1 to boost low intensities

        print(f"Heatmap min: {np.min(heatmap)}, max: {np.max(heatmap)}")  

        return heatmap


def generate_heatmap(heatmap, image_path, output_path):
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt

    # Load original image (BGR)
    original_image = cv2.imread(image_path)
    if original_image is None:
        print(f"❌ Failed to load original image: {image_path}")
        return

    # Normalize heatmap (just in case)
    heatmap = np.maximum(heatmap, 0)
    heatmap = heatmap / (np.max(heatmap) + 1e-8)

    # Resize heatmap to match image size (width, height)
    heatmap_resized = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))

    # Convert heatmap to uint8 (0-255)
    heatmap_uint8 = np.uint8(255 * heatmap_resized)

    # Apply colormap
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    # Overlay heatmap on original image (weights can be tuned)
    superimposed = cv2.addWeighted(heatmap_color, 0.4, original_image, 0.6, 0)

    # Save superimposed image (BGR)
    cv2.imwrite(output_path, superimposed)

    # Also save the raw heatmap as an RGB PNG for debug (small, but with colormap)
    plt.imsave("outputs/debug_heatmap.png", heatmap_resized, cmap='jet')

    print(f"✅ Saved Grad-CAM heatmap to {output_path}")
    

# Transform input image
def preprocess_image(img_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    img = Image.open(img_path).convert("RGB")
    return transform(img).unsqueeze(0)


# Find latest saved model
def find_latest_model(model_dir="saved_models", prefix="resnet18_lc25000"):
    model_files = [os.path.join(model_dir, f) for f in os.listdir(model_dir) if f.startswith(prefix)]
    if not model_files:
        raise FileNotFoundError(f"No model files found in {model_dir} with prefix {prefix}")
    latest_model = max(model_files, key=os.path.getctime)
    print(f"📂 Using latest model: {latest_model}")
    return latest_model


# Main Grad-CAM pipeline
def main(image_path, model_path, layer_name = "layer4.1.conv2", num_classes=5, output_dir="outputs"):
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available()
                          else "cpu")

    model = build_model(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Pick the target layer
    target_layer = dict([*model.named_modules()])[layer_name]

    cam = GradCAM(model, target_layer)
    input_tensor = preprocess_image(image_path).to(device)

    heatmap = cam(input_tensor)
    cam.remove_hooks()

    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.basename(image_path).split('.')[0] + "_gradcam.jpg"
    output_path = os.path.join(output_dir, filename)
    generate_heatmap(heatmap, image_path, output_path)
    print(f"✅ Saved Grad-CAM heatmap to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, required=True, help="Path to test image")
    parser.add_argument('--model_path', type=str, default=None, help="Path to trained model")
    parser.add_argument('--layer_name', type=str, default="layer4.1.conv2", help="Model layer to inspect")
    parser.add_argument('--num_classes', type=int, default=5, help="Number of output classes")
    args = parser.parse_args()

    if args.model_path is None:
        args.model_path = find_latest_model()

    main(args.image_path, args.model_path, args.layer_name, args.num_classes)