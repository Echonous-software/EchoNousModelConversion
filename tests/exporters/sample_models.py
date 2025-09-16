import torch
import torch.nn as nn

from echonous.models import Model, ModuleInputOutput

class MultiIOModel(nn.Module):
    """
    A simple PyTorch model with multiple inputs and outputs for testing conversion pipelines.

    Inputs:
        - vector_input: (N, D) tensor
        - image_input: (N, C, H, W) tensor

    Outputs:
        - class_output_1: (N, X) tensor
        - class_output_2: (N, X) tensor
        - segmentation_output: (N, Y, H, W) tensor
    """

    def __init__(self, D=128, C=3, H=224, W=224, X=10, Y=5):
        super().__init__()

        # Store dimensions
        self.D = D
        self.C = C
        self.H = H
        self.W = W
        self.X = X
        self.Y = Y

        # Simple layers for vector input
        self.vector_fc = nn.Linear(D, X)

        # Simple conv for image input -> classification
        self.image_conv = nn.Conv2d(C, 16, kernel_size=3, padding=1)
        self.image_pool = nn.AdaptiveAvgPool2d(1)
        self.image_fc = nn.Linear(16, X)

        # Simple conv for image input -> segmentation
        self.seg_conv1 = nn.Conv2d(C, 32, kernel_size=3, padding=1)
        self.seg_conv2 = nn.Conv2d(32, Y, kernel_size=3, padding=1)

    def forward(self, vector_input, image_input):
        """
        Args:
            vector_input: (N, D) tensor
            image_input: (N, C, H, W) tensor

        Returns:
            tuple of (class_output_1, class_output_2, segmentation_output)
        """
        # Process vector input -> first classification output
        class_output_1 = self.vector_fc(vector_input)

        # Process image input -> second classification output
        x = self.image_conv(image_input)
        x = torch.relu(x)
        x = self.image_pool(x)
        x = x.flatten(1)
        class_output_2 = self.image_fc(x)

        # Process image input -> segmentation output
        seg = self.seg_conv1(image_input)
        seg = torch.relu(seg)
        segmentation_output = self.seg_conv2(seg)

        return class_output_1, class_output_2, segmentation_output

    def definition(self):
        return Model(
            name="MultiIOModel",
            version="1.0",
            description=MultiIOModel.__doc__,
            inputs=[
                ModuleInputOutput(name="vector_input", shape=(1, self.D)),
                ModuleInputOutput(name="image_input", shape=(1, self.C, self.H, self.W), type="image", scale=1.0 / 255.0),
            ],
            outputs=[
                ModuleInputOutput(name="class_output_1", shape=(1, self.X)),
                ModuleInputOutput(name="class_output_2", shape=(1, self.X)),
                ModuleInputOutput(name="segmentation_output", shape=(1, self.Y, self.H, self.W)),
            ],
            module=self
        )
