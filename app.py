import streamlit as st
import torch, torchvision.utils as vutils
from PIL import Image
from train_cgan import Generator  # usa la misma clase que en el training

LATENT = 100
DEVICE = "cpu"

@st.cache_resource
def load_generator():
    G = Generator()
    G.load_state_dict(torch.load("generator_cgan.pth", map_location=DEVICE))
    G.eval()
    return G

G = load_generator()

st.title("Generador de dígitos manuscritos")
digit = st.selectbox("Elige un dígito (0-9)", list(range(10)))
if st.button("Generar 5 imágenes"):
    z = torch.randn(5, LATENT)
    labels = torch.full((5,), int(digit), dtype=torch.long)
    with torch.no_grad():
        imgs = G(z, labels).cpu()
    grid = vutils.make_grid(imgs, nrow=5, normalize=True, pad_value=1)
    st.image(grid.permute(1,2,0).numpy(), caption=f"Dígito {digit}")