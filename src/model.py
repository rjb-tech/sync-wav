import os
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import torchvision
from torchvision import transforms

LATENT_SIZE = 735
IMAGE_SIZE = 150
BATCH_SIZE = 10
DEV = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def preprocess(x, y):
    return x.view(-1, 3, IMAGE_SIZE, IMAGE_SIZE).to(DEV), y.to(DEV)


class WrappedDataLoader:
    def __init__(self, dl, func):
        self.dl = dl
        self.func = func

    def __len__(self):
        return len(self.dl)

    def __iter__(self):
        batches = iter(self.dl)
        for b in batches:
            yield (self.func(*b))


class Discriminator(nn.Module):
    def __init__(self, img_shape=(3, IMAGE_SIZE, IMAGE_SIZE)):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(img_shape[0], 32, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ZeroPad2d((0, 1, 0, 1)),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            nn.Conv2d(128, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            nn.Conv2d(256, 512, 3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, 1024, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(1024, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        validity = self.model(img)
        return validity.view(-1, 1).squeeze(1)


class Generator(nn.Module):
    def __init__(self, latent_dim=LATENT_SIZE, img_shape=(3, IMAGE_SIZE, IMAGE_SIZE)):
        super(Generator, self).__init__()

        self.init_size = img_shape[2] // 4  # Initial size before upsampling
        self.l1 = nn.Sequential(nn.Linear(latent_dim, 128 * self.init_size**2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, img_shape[0], 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


def train_gan(
    generator,
    discriminator,
    real_imgs,
    device,
    num_epochs,
    batch_size,
    print_every=100,
    latent_dim=LATENT_SIZE,
):
    generator.to(DEV)
    discriminator.to(DEV)

    # Initialize the optimizers and loss function
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0004)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0002)
    criterion = torch.nn.BCELoss()

    disc_loss_by_epoch = []
    gen_loss_by_epoch = []

    # Start training
    for epoch in tqdm(range(num_epochs)):
        for real_batch, _ in real_imgs:
            # Reset gradients
            optimizer_G.zero_grad()
            optimizer_D.zero_grad()

            # Generate fake images
            z = torch.randn(batch_size, latent_dim).to(device)
            fake_imgs = generator.forward(z)

            # Train the discriminator on real images
            validity_real = discriminator.forward(real_batch)
            loss_real = criterion(validity_real, torch.ones_like(validity_real))

            # Train the discriminator on fake images
            validity_fake = discriminator(fake_imgs.detach())
            loss_fake = criterion(validity_fake, torch.zeros_like(validity_fake))

            # Compute the discriminator loss
            d_loss = (loss_real + loss_fake) / 2

            # Backpropagate the discriminator loss
            d_loss.backward()
            optimizer_D.step()

            # Train the generator
            z = torch.randn(batch_size, latent_dim).to(device)
            fake_imgs = generator(z)
            validity_fake = discriminator(fake_imgs)
            g_loss = criterion(validity_fake, torch.ones_like(validity_fake))

            # Backpropagate the generator loss
            g_loss.backward()
            optimizer_G.step()

            # # Print loss every `print_every` iterations
            # if (i + 1) % print_every == 0:
            #     print(
            #         "Epoch [{}/{}], Step [{}/{}], D_Loss: {:.4f}, G_Loss: {:.4f}".format(
            #             epoch + 1,
            #             num_epochs,
            #             i + 1,
            #             len(real_imgs),
            #             d_loss.item(),
            #             g_loss.item(),
            #         )
            #
            #     )

        disc_loss_by_epoch.append(d_loss.item())
        gen_loss_by_epoch.append(g_loss.item())

    return {
        "generator": {
            "state_dict": generator.state_dict(),
            "optimizer": optimizer_G.state_dict(),
            "loss": gen_loss_by_epoch,
        },
        "discriminator": {
            "state_dict": discriminator.state_dict(),
            "optimizer": optimizer_D.state_dict(),
            "loss": disc_loss_by_epoch,
        },
        "epoch": num_epochs,
    }


def main():
    # Define the transform to apply to the images
    transform = transforms.Compose(
        [
            # transforms.Resize(64),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    project_dir = os.getcwd()
    data_folder = os.path.join(project_dir, "data", "images")

    # Load the dataset
    dataset = torchvision.datasets.ImageFolder(root=data_folder, transform=transform)

    # Define the dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4
    )

    gpu_loaded_data = WrappedDataLoader(dataloader, preprocess)
    gen = Generator()
    disc = Discriminator()

    gan = train_gan(
        generator=gen,
        discriminator=disc,
        real_imgs=gpu_loaded_data,
        num_epochs=25,
        device=DEV,
        batch_size=BATCH_SIZE,
    )

    gen.to(DEV)
    z = torch.randn(1, LATENT_SIZE).to(DEV)
    x = gen.forward(z)

    plt.imshow(transforms.ToPILImage()(x[0]), interpolation="bilinear")
    plt.show()

    to_save = input("Would you like to save the model state? ")

    if to_save.lower() == "y" or to_save.lower() == "yes":
        save_name = input("State savename: ")
        save_path = os.path.join(project_dir, "models", save_name)

        torch.save(gan, save_path)


if __name__ == "__main__":
    main()
