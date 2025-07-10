from torchinfo import summary
import argparse
from scripts.model import build_model
import torch

def main(num_classes=5, input_size=(1, 3, 224, 224)):
    model = build_model(num_classes=num_classes)
    summary_str = summary(
        model,
        input_size=input_size,
        col_names=["input_size", "output_size", "num_params", "kernel_size", "mult_adds"],
        depth=3,
        verbose=1
    )
    print(summary_str)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Print a summary of the model architecture.")
    parser.add_argument('--num_classes', type=int, default=5, help='Number of output classes')
    parser.add_argument('--input_size', type=int, nargs=4, default=[1, 3, 224, 224], help='Input size as 4 integers: batch, channels, height, width')
    args = parser.parse_args()
    main(num_classes=args.num_classes, input_size=tuple(args.input_size)) 