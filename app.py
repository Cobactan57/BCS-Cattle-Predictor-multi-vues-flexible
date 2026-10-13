import io

import numpy as np
import streamlit as st
import torch
from PIL import Image, ImageOps
from supabase import create_client
from torch import nn
from torchvision import models, transforms

MODEL_BUCKET = "bcs-models"
MODEL_OBJECT = "models/bcs_mobilenet_v3_flexible.pt"

VIEW_ORDER = ["Dorso-caudale", "Arrière", "Latérale gauche", "Latérale droite"]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

st.set_page_config(page_title="BCS Predictor — Flexible multi-vues", page_icon="🐄", layout="wide")


def _secret(*names):
    for name in names:
        value = st.secrets.get(name, "")
        if value:
            return value
    raise RuntimeError("Secret manquant : " + " ou ".join(names))


@st.cache_resource
def get_supabase():
    return create_client(
        _secret("SUPABASE_URL"),
        _secret("SUPABASE_SECRET_KEY", "SUPABASE_KEY"),
    )


class FlexibleMultiViewMobileNet(nn.Module):
    def __init__(self):
        super().__init__()
        base = models.mobilenet_v3_small(weights=None)
        self.features = base.features
        self.avgpool = base.avgpool
        self.feature_dim = base.classifier[0].in_features
        self.regressor = nn.Sequential(
            nn.Linear((self.feature_dim + 1) * len(VIEW_ORDER), 512),
            nn.Hardswish(),
            nn.Dropout(0.25),
            nn.Linear(512, 128),
            nn.Hardswish(),
            nn.Dropout(0.15),
            nn.Linear(128, 1),
        )

    def encode(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        return torch.flatten(x, 1)

    def forward(self, x, mask):
        b, v, c, h, w = x.shape
        f = self.encode(x.reshape(b * v, c, h, w))
        f = f.reshape(b, v, self.feature_dim)
        f = f * mask.unsqueeze(-1)
        z = torch.cat([f, mask.unsqueeze(-1)], dim=-1).reshape(b, -1)
        return self.regressor(z).squeeze(1)


@st.cache_resource(show_spinner="Chargement du modèle flexible…")
def load_predictor():
    payload = get_supabase().storage.from_(MODEL_BUCKET).download(MODEL_OBJECT)
    bundle = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=False)

    if bundle.get("architecture") != "mobilenet_v3_small_flexible_multiview":
        raise RuntimeError("Le fichier de modèle n'est pas le modèle flexible attendu.")

    model = FlexibleMultiViewMobileNet()
    model.load_state_dict(bundle["state_dict"])
    model.eval()

    tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(int(bundle.get("input_size", 224))),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return model, tf, bundle


def round_quarter(x):
    return round(x * 4) / 4


def predict(images_by_view, model, tf, bundle):
    tensors = []
    mask = []

    blank = torch.zeros(3, int(bundle.get("input_size", 224)), int(bundle.get("input_size", 224)))
    for view in VIEW_ORDER:
        img = images_by_view.get(view)
        if img is None:
            tensors.append(blank)
            mask.append(0.0)
        else:
            tensors.append(tf(img.convert("RGB")))
            mask.append(1.0)

    x = torch.stack(tensors).unsqueeze(0)
    m = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        raw = float(model(x, m).item())

    raw = float(np.clip(raw, bundle.get("score_min", 1.0), bundle.get("score_max", 5.0)))
    return raw, round_quarter(raw), int(sum(mask))


st.title("🐄 BCS Cattle Predictor — multi-vues flexible")
st.caption("Le modèle accepte de 1 à 4 vues d'un même animal.")

try:
    model, tf, bundle = load_predictor()
except Exception as exc:
    st.error("Le modèle flexible ne peut pas être chargé.")
    st.code(str(exc))
    st.stop()

# Une seule source active par vue : import depuis la galerie ou prise de vue directe.
# Les widgets conservent leurs images pendant les interactions Streamlit.
uploads = {}
st.subheader("Acquisition des photographies")
st.caption("Pour chaque vue, importez une image ou photographiez directement l'animal. Une à quatre vues suffisent.")

for idx, view in enumerate(VIEW_ORDER):
    with st.expander(f"{idx + 1}. {view}", expanded=(idx == 0)):
        source = st.radio(
            f"Source de l'image — {view}",
            options=("Prendre une photo", "Importer une photo"),
            key=f"source_{idx}",
            horizontal=True,
        )

        if source == "Prendre une photo":
            f = st.camera_input(
                f"Photographier : {view}",
                key=f"camera_{idx}",
            )
        else:
            f = st.file_uploader(
                f"Importer : {view}",
                type=["jpg", "jpeg", "png"],
                key=f"upload_{idx}",
            )

        if f is not None:
            try:
                # Corrige l'orientation EXIF des images prises avec un téléphone.
                with Image.open(f) as image:
                    img = ImageOps.exif_transpose(image).convert("RGB")
                uploads[view] = img
                st.image(img, caption=f"Vue sélectionnée : {view}", use_container_width=True)
            except (OSError, ValueError) as exc:
                st.error(f"Image illisible pour la vue {view} : {exc}")

st.caption(f"{len(uploads)} vue(s) prête(s) pour la prédiction sur 4 possibles.")

n = len(uploads)

if n == 0:
    st.info("Importez ou prenez au moins une photo.")
else:
    if n == 4:
        st.success("4 vues disponibles : configuration optimale.")
    elif n == 3:
        st.info("3 vues disponibles : prédiction multi-vues partielle.")
    elif n == 2:
        st.warning("2 vues disponibles : précision potentiellement réduite.")
    else:
        st.warning("1 seule vue : estimation indicative, moins robuste.")

    if st.button("Déterminer le BCS", type="primary", use_container_width=True):
        raw, rounded, n_used = predict(uploads, model, tf, bundle)
        st.metric("BCS estimé", f"{rounded:.2f}")
        st.caption(f"Prédiction continue : {raw:.2f} — {n_used} vue(s) utilisée(s)")

        metrics = bundle.get("validation_metrics_by_n_views", {}).get(str(n_used), {})
        if metrics:
            c1, c2, c3 = st.columns(3)
            c1.metric("MAE validation", f"{metrics.get('mae', float('nan')):.3f}")
            c2.metric("±0,25", f"{100*metrics.get('within_0_25', 0):.1f} %")
            c3.metric("±0,50", f"{100*metrics.get('within_0_50', 0):.1f} %")

            q95 = metrics.get("residual_abs_q95")
            if q95 is not None:
                lo = max(bundle.get("score_min", 1.0), raw - q95)
                hi = min(bundle.get("score_max", 5.0), raw + q95)
                st.write(f"Intervalle empirique 95 % : **{lo:.2f} à {hi:.2f}**")

with st.expander("Performances selon le nombre de vues"):
    perf = bundle.get("validation_metrics_by_n_views", {})
    for k in ["4", "3", "2", "1"]:
        if k in perf:
            m = perf[k]
            st.write(
                f"**{k} vue(s)** — MAE {m['mae']:.3f} | "
                f"±0,25 : {100*m['within_0_25']:.1f} % | "
                f"±0,50 : {100*m['within_0_50']:.1f} %"
            )

st.warning(
    "Le score est algorithmique. Les performances doivent être validées sur des animaux indépendants "
    "avant toute utilisation scientifique ou clinique."
)
