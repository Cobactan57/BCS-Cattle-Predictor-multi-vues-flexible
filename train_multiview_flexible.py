import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit
from supabase import create_client
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.models import MobileNet_V3_Small_Weights

PHOTO_BUCKET = os.getenv("BCS_PHOTO_BUCKET", "bcs-photos")
MODEL_BUCKET = os.getenv("BCS_MODEL_BUCKET", "bcs-models")
MODEL_OBJECT = os.getenv("BCS_MODEL_OBJECT", "models/bcs_mobilenet_v3_flexible.pt")
LOCAL_MODEL = Path(os.getenv("BCS_LOCAL_MODEL", "bcs_mobilenet_v3_flexible.pt"))
CACHE_DIR = Path(os.getenv("BCS_CACHE_DIR", ".cache_bcs_photos"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)

SEED = int(os.getenv("BCS_SEED", "42"))
BATCH_SIZE = int(os.getenv("BCS_BATCH_SIZE", "4"))
FROZEN_EPOCHS = int(os.getenv("BCS_FROZEN_EPOCHS", "6"))
FINETUNE_EPOCHS = int(os.getenv("BCS_FINETUNE_EPOCHS", "18"))
MIN_REFERENCE_SCORES_PER_VIEW = int(os.getenv("BCS_MIN_REFERENCE_SCORES_PER_VIEW", "2"))
VAL_FRACTION = float(os.getenv("BCS_VAL_FRACTION", "0.20"))
VIEW_DROPOUT_PROB = float(os.getenv("BCS_VIEW_DROPOUT_PROB", "0.70"))
INPUT_SIZE = 224

VIEW_ORDER = ["Dorso-caudale", "Arrière", "Latérale gauche", "Latérale droite"]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_secret(name: str, fallback: str | None = None) -> str:
    value = os.getenv(name)
    if value:
        return value
    try:
        import tomllib
        p = Path(".streamlit/secrets.toml")
        if p.exists():
            data = tomllib.loads(p.read_text(encoding="utf-8"))
            if name in data:
                return str(data[name])
            if fallback and fallback in data:
                return str(data[fallback])
    except Exception:
        pass
    if fallback:
        value = os.getenv(fallback)
        if value:
            return value
    raise RuntimeError(f"Secret manquant: {name}" + (f" (ou {fallback})" if fallback else ""))


def make_supabase():
    return create_client(
        get_secret("SUPABASE_URL"),
        get_secret("SUPABASE_SECRET_KEY", "SUPABASE_KEY"),
    )


def _slug(view: str) -> str:
    return (
        view.lower()
        .replace("é", "e").replace("è", "e").replace("à", "a")
        .replace(" ", "_").replace("-", "_")
    )


def fetch_sessions(sb) -> pd.DataFrame:
    acq = pd.DataFrame(
        sb.table("acquisitions")
        .select(
            "acquisition_id,animal_id,date_acquisition,vue_photo,photo_path,quality_status,"
            "bcs_operateur_1,bcs_operateur_2,bcs_operateur_3"
        )
        .eq("quality_status", "accepted")
        .execute().data or []
    )
    if acq.empty:
        raise RuntimeError("Aucune acquisition accepted trouvée.")

    scores = pd.DataFrame(
        sb.table("scores").select("acquisition_id,bcs,evaluateur").execute().data or []
    )
    independent = {}
    if not scores.empty:
        independent = scores.groupby("acquisition_id")["bcs"].apply(list).to_dict()

    rows = []
    for _, r in acq.iterrows():
        view = r.get("vue_photo")
        if view not in VIEW_ORDER:
            continue
        refs = []
        for c in ["bcs_operateur_1", "bcs_operateur_2", "bcs_operateur_3"]:
            v = r.get(c)
            if pd.notna(v):
                refs.append(float(v))
        refs += [float(x) for x in independent.get(r["acquisition_id"], []) if pd.notna(x)]
        refs = [x for x in refs if 1.0 <= x <= 5.0]
        if len(refs) < MIN_REFERENCE_SCORES_PER_VIEW:
            continue
        rows.append({
            "acquisition_id": str(r["acquisition_id"]),
            "animal_id": str(r["animal_id"]),
            "date_acquisition": str(r["date_acquisition"]),
            "vue_photo": str(view),
            "photo_path": str(r["photo_path"]),
            "refs": refs,
        })

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("Aucune donnée exploitable après filtrage.")

    df["session_id"] = df["animal_id"] + "__" + df["date_acquisition"]
    sessions = []

    for session_id, g in df.groupby("session_id"):
        if any((g["vue_photo"] == v).sum() != 1 for v in VIEW_ORDER):
            continue
        g = g.set_index("vue_photo")
        all_refs = []
        for v in VIEW_ORDER:
            all_refs += list(g.loc[v, "refs"])
        row = {
            "session_id": session_id,
            "animal_id": str(g.iloc[0]["animal_id"]),
            "date_acquisition": str(g.iloc[0]["date_acquisition"]),
            "bcs_reference": float(np.median(all_refs)),
        }
        for v in VIEW_ORDER:
            slug = _slug(v)
            row[f"path_{slug}"] = g.loc[v, "photo_path"]
            row[f"acq_{slug}"] = g.loc[v, "acquisition_id"]
        sessions.append(row)

    out = pd.DataFrame(sessions)
    if len(out) < 10:
        raise RuntimeError(f"Seulement {len(out)} sessions complètes à 4 vues.")
    return out


def cache_photos(sb, df):
    valid = []
    for _, row in df.iterrows():
        d = row.to_dict()
        ok = True
        for v in VIEW_ORDER:
            slug = _slug(v)
            remote = row[f"path_{slug}"]
            acq_id = row[f"acq_{slug}"]
            ext = Path(remote).suffix.lower() or ".jpg"
            local = CACHE_DIR / f"{acq_id}{ext}"
            if not local.exists():
                try:
                    local.write_bytes(sb.storage.from_(PHOTO_BUCKET).download(remote))
                except Exception as exc:
                    print(f"Download failed {remote}: {exc}")
                    ok = False
                    break
            try:
                with Image.open(local) as im:
                    im.verify()
                d[f"local_{slug}"] = str(local)
            except Exception as exc:
                print(f"Invalid image {local}: {exc}")
                ok = False
                break
        if ok:
            valid.append(d)
    out = pd.DataFrame(valid)
    if out.empty:
        raise RuntimeError("Aucune session complète avec 4 images valides.")
    return out


def make_transforms():
    norm = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(INPUT_SIZE, scale=(0.84, 1.0), ratio=(0.92, 1.08)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(6),
        transforms.ColorJitter(brightness=0.10, contrast=0.10, saturation=0.06),
        transforms.ToTensor(),
        norm,
    ])
    val_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(INPUT_SIZE),
        transforms.ToTensor(),
        norm,
    ])
    return train_tf, val_tf


class FlexibleMultiViewDataset(Dataset):
    def __init__(self, frame, transform, training=False):
        self.frame = frame.reset_index(drop=True)
        self.transform = transform
        self.training = training

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        row = self.frame.iloc[idx]
        imgs = []
        for v in VIEW_ORDER:
            slug = _slug(v)
            with Image.open(row[f"local_{slug}"]) as im:
                imgs.append(self.transform(im.convert("RGB")))
        x = torch.stack(imgs)  # [4,C,H,W]

        mask = torch.ones(len(VIEW_ORDER), dtype=torch.float32)
        if self.training and np.random.rand() < VIEW_DROPOUT_PROB:
            # Garde aléatoirement 1 à 3 vues; parfois les 4 restent présentes.
            keep_n = np.random.randint(1, len(VIEW_ORDER))
            keep = np.random.choice(len(VIEW_ORDER), size=keep_n, replace=False)
            mask[:] = 0.0
            mask[keep] = 1.0

        y = torch.tensor(float(row["bcs_reference"]), dtype=torch.float32)
        return x, mask, y


class FlexibleMultiViewMobileNet(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        base = models.mobilenet_v3_small(weights=weights)
        self.features = base.features
        self.avgpool = base.avgpool
        self.feature_dim = base.classifier[0].in_features

        # Chaque vue produit features + présence/absence explicite.
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

        # Annule les caractéristiques des vues absentes.
        f = f * mask.unsqueeze(-1)
        # Ajoute explicitement le masque comme feature.
        z = torch.cat([f, mask.unsqueeze(-1)], dim=-1)
        z = z.reshape(b, -1)
        return self.regressor(z).squeeze(1)


def run_epoch(model, loader, loss_fn, optimizer, device, train):
    model.train(train)
    total, n = 0.0, 0
    for x, mask, y in loader:
        x, mask, y = x.to(device), mask.to(device), y.to(device)
        if train:
            optimizer.zero_grad(set_to_none=True)
        pred = model(x, mask)
        loss = loss_fn(pred, y)
        if train:
            loss.backward()
            optimizer.step()
        total += float(loss.item()) * len(y)
        n += len(y)
    return total / max(n, 1)


@torch.no_grad()
def eval_with_n_views(model, frame, transform, device, n_views):
    ds = FlexibleMultiViewDataset(frame, transform, training=False)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    ys, ps = [], []
    rng = np.random.default_rng(SEED + n_views)
    for x, mask, y in loader:
        batch = len(y)
        test_mask = torch.zeros(batch, len(VIEW_ORDER), dtype=torch.float32)
        for i in range(batch):
            keep = rng.choice(len(VIEW_ORDER), size=n_views, replace=False)
            test_mask[i, keep] = 1.0
        pred = model(x.to(device), test_mask.to(device)).cpu().numpy()
        ys.extend(y.numpy().tolist())
        ps.extend(pred.tolist())

    y = np.asarray(ys, dtype=float)
    p = np.clip(np.asarray(ps, dtype=float), 1.0, 5.0)
    ae = np.abs(p - y)
    return {
        "mae": float(ae.mean()),
        "rmse": float(np.sqrt(np.mean((p - y) ** 2))),
        "within_0_25": float(np.mean(ae <= 0.25)),
        "within_0_50": float(np.mean(ae <= 0.50)),
        "residual_abs_q95": float(np.quantile(ae, 0.95)),
    }


def train():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    sb = make_supabase()
    df = cache_photos(sb, fetch_sessions(sb))

    splitter = GroupShuffleSplit(n_splits=1, test_size=VAL_FRACTION, random_state=SEED)
    train_idx, val_idx = next(splitter.split(df, groups=df["animal_id"]))
    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)

    overlap = set(train_df["animal_id"]) & set(val_df["animal_id"])
    if overlap:
        raise RuntimeError("Fuite de données entre entraînement et validation.")

    train_tf, val_tf = make_transforms()
    train_loader = DataLoader(
        FlexibleMultiViewDataset(train_df, train_tf, training=True),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        FlexibleMultiViewDataset(val_df, val_tf, training=False),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Sessions: {len(df)} | Animaux: {df['animal_id'].nunique()}")
    print(f"Train: {len(train_df)} | Validation: {len(val_df)}")

    model = FlexibleMultiViewMobileNet(pretrained=True).to(device)
    loss_fn = nn.SmoothL1Loss(beta=0.25)

    for p in model.features.parameters():
        p.requires_grad = False

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=8e-4, weight_decay=1e-4
    )

    for epoch in range(FROZEN_EPOCHS):
        tr = run_epoch(model, train_loader, loss_fn, optimizer, device, True)
        va = run_epoch(model, val_loader, loss_fn, optimizer, device, False)
        print(f"Frozen {epoch+1}/{FROZEN_EPOCHS}: train={tr:.4f} val={va:.4f}")

    for p in model.features.parameters():
        p.requires_grad = True

    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-5, weight_decay=1e-4)

    best_state = None
    best_mae = math.inf
    bad_epochs = 0
    patience = 5

    for epoch in range(FINETUNE_EPOCHS):
        tr = run_epoch(model, train_loader, loss_fn, optimizer, device, True)
        va = run_epoch(model, val_loader, loss_fn, optimizer, device, False)
        m4 = eval_with_n_views(model, val_df, val_tf, device, 4)
        print(f"FT {epoch+1}/{FINETUNE_EPOCHS}: train={tr:.4f} val={va:.4f} MAE4={m4['mae']:.3f}")

        if m4["mae"] < best_mae:
            best_mae = m4["mae"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print("Early stopping")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model = model.to(device)

    metrics_by_views = {
        str(n): eval_with_n_views(model, val_df, val_tf, device, n)
        for n in [1, 2, 3, 4]
    }

    bundle = {
        "architecture": "mobilenet_v3_small_flexible_multiview",
        "view_order": VIEW_ORDER,
        "state_dict": model.cpu().state_dict(),
        "input_size": INPUT_SIZE,
        "score_min": 1.0,
        "score_max": 5.0,
        "score_increment": 0.25,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_sessions_total": int(len(df)),
        "n_sessions_train": int(len(train_df)),
        "n_sessions_validation": int(len(val_df)),
        "n_animals_total": int(df["animal_id"].nunique()),
        "view_dropout_probability": VIEW_DROPOUT_PROB,
        "validation_metrics_by_n_views": metrics_by_views,
    }

    torch.save(bundle, LOCAL_MODEL)
    meta = LOCAL_MODEL.with_suffix(".json")
    meta.write_text(
        json.dumps({k: v for k, v in bundle.items() if k != "state_dict"},
                   indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    for remote, local in [(MODEL_OBJECT, LOCAL_MODEL), (MODEL_OBJECT + ".json", meta)]:
        try:
            sb.storage.from_(MODEL_BUCKET).remove([remote])
        except Exception:
            pass
        sb.storage.from_(MODEL_BUCKET).upload(
            path=remote,
            file=local.read_bytes(),
            file_options={"content-type": "application/octet-stream", "upsert": "true"},
        )

    print("\nTraining complete")
    print(json.dumps(metrics_by_views, indent=2))
    print(f"Model saved locally: {LOCAL_MODEL}")
    print(f"Model uploaded: {MODEL_BUCKET}/{MODEL_OBJECT}")


if __name__ == "__main__":
    train()
