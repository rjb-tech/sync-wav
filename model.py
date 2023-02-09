import torch.nn as nn
import torch
import matplotlib.pyplot as plt
from torchvision import transforms

LATENT_SIZE = 735
IMAGE_SIZE = 64
BATCH_SIZE = 64


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.latent_upscale = torch.nn.ConvTranspose2d(
            LATENT_SIZE, IMAGE_SIZE * 16, kernel_size=4, bias=False
        )
        self.latent_norm = torch.nn.BatchNorm2d(IMAGE_SIZE * 16)
        self.latent_activation = torch.nn.ReLU()

        self.layer_1 = nn.ConvTranspose2d(
            IMAGE_SIZE * 16, IMAGE_SIZE * 8, kernel_size=2, stride=2, bias=False
        )
        self.layer_1_norm = torch.nn.BatchNorm2d(IMAGE_SIZE * 8)
        self.layer_1_dropout = nn.Dropout2d(0.5)
        self.layer_1_activation = torch.nn.ReLU()

        self.layer_2 = torch.nn.ConvTranspose2d(
            IMAGE_SIZE * 8, IMAGE_SIZE * 4, kernel_size=2, stride=2, bias=False
        )
        self.layer_2_norm = torch.nn.BatchNorm2d(IMAGE_SIZE * 4)
        self.layer_2_dropout = nn.Dropout2d(0.5)
        self.layer_2_activation = torch.nn.ReLU()
        torch.nn.init.xavier_uniform_(self.layer_2.weight)

        self.layer_3 = torch.nn.PixelShuffle(2)
        self.layer_3_norm = torch.nn.BatchNorm2d(IMAGE_SIZE)
        self.layer_3_dropout = nn.Dropout2d(0.5)
        self.layer_3_activation = torch.nn.ReLU()

        self.layer_4 = torch.nn.ConvTranspose2d(
            IMAGE_SIZE, 3, kernel_size=2, stride=2, bias=False
        )
        torch.nn.init.xavier_uniform_(self.layer_4.weight)
        self.layer_4_norm = nn.BatchNorm2d(3)
        self.layer_4_dropout = nn.Dropout2d(0.5)
        self.layer_4_activation = torch.nn.Tanh()

    def forward(self, x):
        x = self.latent_upscale(x)
        x = self.latent_norm(x)
        x = self.latent_activation(x)
        x = self.layer_1(x)
        # x = self.layer_1_norm(x)
        x = self.layer_1_dropout(x)
        x = self.layer_1_activation(x)
        x = self.layer_2(x)
        # x = self.layer_2_norm(x)
        x = self.layer_2_dropout(x)
        x = self.layer_2_activation(x)
        x = self.layer_3(x)
        # x = self.layer_3_norm(x)
        x = self.layer_3_dropout(x)
        x = self.layer_3_activation(x)
        x = self.layer_4(x)
        x = self.layer_4_norm(x)
        x = self.layer_4_activation(x)

        return x


class Discriminator:
    pass


def main():
    model = Generator()
    x = torch.randn(1, LATENT_SIZE, 1, 1)
    y = model.forward(x)

    plt.imshow(transforms.ToPILImage()(y[0]), interpolation="bilinear")
    plt.show()


if __name__ == "__main__":
    main()
