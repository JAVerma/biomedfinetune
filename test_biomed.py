import os, glob, shutil, torch, torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from tqdm import tqdm
import open_clip
from sklearn.metrics import accuracy_score
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# ---------------- CONFIG ----------------------------------
TEST_DIR        = "/home/jayant/Desh4/pasted_Test"
CKPT_PATTERN    = "/home/jayant/Desh4/finetune/*.pth"
MODEL_ID        = "hf-hub:microsoft/BioMedCLIP-PubMedBERT_256-vit_base_patch16_224"
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE      = 1
LABEL_NAMES     = ["non-desh", "Desh"]             # ImageFolder alphabetical order
MISCLS_ROOT     = "/home/jayant/Desh4/misclassified"
os.makedirs(MISCLS_ROOT, exist_ok=True)

# --------------- DATASET ----------------------------------
mean = [0.48145466, 0.4578275, 0.40821073]
std  = [0.26862954, 0.26130258, 0.27577711]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

test_ds  = ImageFolder(TEST_DIR, transform=transform)
test_dl  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
# keep original file paths
file_paths = [p for p, _ in test_ds.samples]

# --------------- MODEL WRAPPER ----------------------------
class BioMedClipClassifier(nn.Module):
    def __init__(self, clip_model, num_classes=2, n_train_blocks=3):
        super().__init__()
        vit            = clip_model.visual.trunk
        self.vit       = vit
        self.embed_dim = vit.embed_dim
        vit.blocks     = vit.blocks[:-1]

        self.bilinear   = nn.Bilinear(self.embed_dim, self.embed_dim, self.embed_dim)
        self.classifier = nn.Linear(self.embed_dim, num_classes)

    def forward(self, x):
        vit = self.vit
        x   = vit.patch_embed(x)
        cls = vit.cls_token.expand(x.size(0), -1, -1)
        x   = vit.pos_drop(torch.cat((cls, x), 1) + vit.pos_embed)

        cls_tokens = []
        for i, blk in enumerate(vit.blocks):
            x = blk(x)
            if i in [len(vit.blocks)-3, len(vit.blocks)-2]:
                cls_tokens.append(x[:, 0])

        pooled = self.bilinear(*cls_tokens)
        pooled = vit.norm(pooled)
        return self.classifier(pooled)

# --------------- Helper: evaluation -----------------------
def evaluate(model):
    preds, gts = [], []
    with torch.no_grad():
        for imgs, labels in tqdm(test_dl, desc="Batches", leave=False):
            imgs = imgs.to(DEVICE)
            out  = model(imgs).argmax(1).cpu()
            preds.extend(out.tolist())
            gts.extend(labels.tolist())
    return preds, gts

# --------------- Iterate checkpoints ----------------------
best_acc, best_ckpt, best_preds = 0.0, None, None
checkpoints = sorted(glob.glob(CKPT_PATTERN))
if not checkpoints:
    raise FileNotFoundError("No checkpoints found!")

for ckpt in checkpoints:
    base, _ = open_clip.create_model_from_pretrained(MODEL_ID, device=DEVICE)
    model   = BioMedClipClassifier(base, len(LABEL_NAMES)).to(DEVICE)
    model.load_state_dict(torch.load(ckpt, map_location="cpu"), strict=True)
    model.eval(); model.requires_grad_(False)

    preds, gts = evaluate(model)
    acc = accuracy_score(gts, preds)
    print(f"{os.path.basename(ckpt):40s} → acc {acc:.4f}")

    if acc > best_acc:
        best_acc, best_ckpt, best_preds = acc, ckpt, preds

print(f"\n🏆 Best checkpoint: {os.path.basename(best_ckpt)}  (acc {best_acc:.4f})")

# --------------- Save mis-classified images ---------------
print("\nSaving misclassified images…")
# reload best model (already in memory if last, but safe)
base,_ = open_clip.create_model_from_pretrained(MODEL_ID, device=DEVICE)
best_model = BioMedClipClassifier(base, len(LABEL_NAMES)).to(DEVICE)
best_model.load_state_dict(torch.load(best_ckpt, map_location="cpu"), strict=True)
best_model.eval(); best_model.requires_grad_(False)

# ensure clean dir
shutil.rmtree(MISCLS_ROOT, ignore_errors=True)
os.makedirs(MISCLS_ROOT, exist_ok=True)

# iterate file-by-file to capture original images & predictions
with torch.no_grad():
    for idx in tqdm(range(len(test_ds)), desc="Images"):
        img_path = file_paths[idx]
        label    = test_ds.targets[idx]               # ground-truth int
        img      = Image.open(img_path).convert("RGB")

        # forward single image
        tensor   = transform(img).unsqueeze(0).to(DEVICE)
        pred_idx = best_model(tensor).argmax(1).item()

        if pred_idx != label:                         # mis-classified
            gt_name   = LABEL_NAMES[label]
            pred_name = LABEL_NAMES[pred_idx]
            save_dir  = os.path.join(MISCLS_ROOT, f"{gt_name}_as_{pred_name}")
            os.makedirs(save_dir, exist_ok=True)
            # annotate
            ann_img = img.copy()
            draw    = ImageDraw.Draw(ann_img)
            banner  = f"GT: {gt_name}  |  PRED: {pred_name}"
            # choose banner height 30px, black bg, yellow text
            draw.rectangle([(0,0),(ann_img.width,30)], fill=(0,0,0))
            draw.text((5,5), banner, fill=(255,255,0))
            ann_img.save(os.path.join(save_dir, os.path.basename(img_path)))

print(f"\nMisclassified images saved under {MISCLS_ROOT}")
