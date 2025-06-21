## pip install torch torchvision streamlit pillow numpy tqdm

import torch, torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

BATCH = 128
LATENT = 100
EPOCHS = 30
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------- Datos ----------
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])
trainset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
loader   = DataLoader(trainset, batch_size=BATCH, shuffle=True)

# ---------- Redes ----------
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.label_emb = nn.Embedding(10, 10)
        self.gen = nn.Sequential(
            nn.Linear(LATENT+10, 256),
            nn.LeakyReLU(0.2, True),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, True),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2, True),
            nn.Linear(1024, 28*28),
            nn.Tanh()
        )

    def forward(self, z, labels):
        x = torch.cat([z, self.label_emb(labels)], dim=1)
        img = self.gen(x)
        return img.view(-1, 1, 28, 28)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.label_emb = nn.Embedding(10, 10)
        self.dis = nn.Sequential(
            nn.Linear(28*28+10, 512),
            nn.LeakyReLU(0.2, True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, True),
            nn.Linear(256, 1)
        )

    def forward(self, img, labels):
        x = torch.cat([img.view(img.size(0), -1), self.label_emb(labels)], dim=1)
        return self.dis(x)

G, D = Generator().to(DEVICE), Discriminator().to(DEVICE)
optG = torch.optim.Adam(G.parameters(), 2e-4, betas=(0.5, 0.999))
optD = torch.optim.Adam(D.parameters(), 2e-4, betas=(0.5, 0.999))
criterion = nn.BCEWithLogitsLoss()

if __name__ == "__main__":
    # ---------- Entrenamiento ----------
    for epoch in range(EPOCHS):
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            bs = imgs.size(0)

            # Discriminator
            z = torch.randn(bs, LATENT, device=DEVICE)
            fake_labels = torch.randint(0, 10, (bs,), device=DEVICE)
            fake_imgs = G(z, fake_labels).detach()
            lossD = (criterion(D(imgs, labels), torch.ones(bs,1,device=DEVICE)) +
                    criterion(D(fake_imgs, fake_labels), torch.zeros(bs,1,device=DEVICE))) / 2
            optD.zero_grad(); lossD.backward(); optD.step()

            # Generator
            z = torch.randn(bs, LATENT, device=DEVICE)
            gen_labels = torch.randint(0, 10, (bs,), device=DEVICE)
            gen_imgs = G(z, gen_labels)
            lossG = criterion(D(gen_imgs, gen_labels), torch.ones(bs,1,device=DEVICE))
            optG.zero_grad(); lossG.backward(); optG.step()

        print(f"Epoch {epoch+1}/{EPOCHS}  LossD {lossD.item():.3f}  LossG {lossG.item():.3f}")

    torch.save(G.state_dict(), "generator_cgan.pth")
    print("Modelo guardado 🎉")