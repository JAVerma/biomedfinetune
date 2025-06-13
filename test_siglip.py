import os, glob, json, shutil, torch, torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from PIL import Image, ImageDraw
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from transformers import SiglipProcessor, SiglipModel

# ---------------- CONFIG ----------------------------------
TEST_DIR     = "/home/jayant/Desh4/pasted_Test"
CKPT_PATTERN = "/home/jayant/Desh4/finetune_siglip/siglip_desh_finetune_*.pth"
MODEL_ID     = "google/siglip2-base-patch16-384"
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE   = 4
LABEL_NAMES  = ["non-desh", "Desh"]
MISCLS_ROOT  = "/home/jayant/Desh4/misclassified_siglip"
os.makedirs(MISCLS_ROOT, exist_ok=True)

# ---------------- Processor -------------------------------
processor = SiglipProcessor.from_pretrained(MODEL_ID)
image_mean = processor.image_mean
image_std  = processor.image_std
image_size = processor.size['height']

# ---------------- Dataset ---------------------------------
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=image_mean, std=image_std),
])

test_ds = ImageFolder(TEST_DIR, transform=transform)
test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
file_paths = [p for p, _ in test_ds.samples]

# ---------------- Model Definition ------------------------
class SigLIPClassifier(nn.Module):
    def __init__(self, base: SiglipModel, num_classes=2):
        super().__init__()
        self.vision_model = base.vision_model
        self.hidden_size = self.vision_model.config.hidden_size
        self.transformer = self.vision_model.encoder
        self.vision_model.encoder.layers = self.vision_model.encoder.layers[:-1]
        self.bilinear = nn.Bilinear(self.hidden_size, self.hidden_size, self.hidden_size)
        self.classifier = nn.Linear(self.hidden_size, num_classes)

    def forward(self, pixel_values):
        embedding = self.vision_model.embeddings(pixel_values)
        attention_mask = torch.ones(
            (embedding.shape[0], embedding.shape[1]),
            dtype=torch.bool, device=embedding.device
        )
        cls_tokens = []
        for i, blk in enumerate(self.transformer.layers):
            embedding = blk(embedding, attention_mask=attention_mask)[0]
            if i in [len(self.transformer.layers)-3, len(self.transformer.layers)-2]:
                cls_tokens.append(embedding[:, 0])
        pooled = self.bilinear(*cls_tokens)
        pooled = self.vision_model.post_layernorm(pooled)
        return self.classifier(pooled)

# ---------------- Evaluate a checkpoint -------------------
def evaluate_checkpoint(path):
    base_model = SiglipModel.from_pretrained(MODEL_ID).to(DEVICE)
    model = SigLIPClassifier(base_model, len(LABEL_NAMES)).to(DEVICE)
    model.load_state_dict(torch.load(path, map_location="cpu"), strict=True)
    model.eval(); model.requires_grad_(False)

    preds, gts = [], []
    with torch.no_grad():
        for imgs, labels in tqdm(test_dl, desc=os.path.basename(path), leave=False):
            preds.extend(model(imgs.to(DEVICE)).argmax(1).cpu().tolist())
            gts.extend(labels.tolist())
    return preds, gts, model

# ---------------- MAIN ------------------------------------
checkpoints = sorted(glob.glob(CKPT_PATTERN))
assert checkpoints, "No checkpoints found!"

best_acc, best_ckpt, best_preds, best_gts = 0, None, None, None
for ckpt in checkpoints:
    preds, gts, _ = evaluate_checkpoint(ckpt)
    acc = accuracy_score(gts, preds)
    print(f"{os.path.basename(ckpt):35s} → acc {acc:.4f}")
    if acc > best_acc:
        best_acc, best_ckpt, best_preds, best_gts = acc, ckpt, preds, gts

print(f"\n🏆 Best checkpoint: {os.path.basename(best_ckpt)}  (acc {best_acc:.4f})")

# ---------------- Confusion & Report ----------------------
cm = confusion_matrix(best_gts, best_preds, labels=[0, 1])
print("\nConfusion matrix:")
print("         " + "  ".join(f"{n:>8s}" for n in LABEL_NAMES))
for i, row in enumerate(cm):
    print(f"{LABEL_NAMES[i]:>8s}  " + "  ".join(f"{v:8d}" for v in row))

report_dict = classification_report(best_gts, best_preds, target_names=LABEL_NAMES,
                                    digits=4, output_dict=True)
print("\nClassification report:\n",
      classification_report(best_gts, best_preds, target_names=LABEL_NAMES, digits=4))

metrics = {
    "overall_accuracy": round(best_acc, 4),
    "per_class_accuracy": {
        LABEL_NAMES[i]: round(report_dict[LABEL_NAMES[i]]["recall"], 4)
        for i in range(len(LABEL_NAMES))
    },
    "confusion_matrix": cm.tolist(),
    "classification_report": report_dict
}
json_path = os.path.join(MISCLS_ROOT, "metrics.json")
with open(json_path, "w") as jf:
    json.dump(metrics, jf, indent=2)
print(f"\n📄 metrics.json written to {json_path}")
print(json.dumps(metrics, indent=2))

# ---------------- Save misclassified -----------------------
shutil.rmtree(MISCLS_ROOT, ignore_errors=True)
os.makedirs(MISCLS_ROOT, exist_ok=True)

with torch.no_grad():
    base_model = SiglipModel.from_pretrained(MODEL_ID).to(DEVICE)
    best_model = SigLIPClassifier(base_model, len(LABEL_NAMES)).to(DEVICE)
    best_model.load_state_dict(torch.load(best_ckpt, map_location="cpu"), strict=True)
    best_model.eval(); best_model.requires_grad_(False)

    for idx in tqdm(range(len(test_ds)), desc="Saving miscls"):
        gt, pred = best_gts[idx], best_preds[idx]
        if gt == pred: continue
        gt_name, pred_name = LABEL_NAMES[gt], LABEL_NAMES[pred]
        out_dir = os.path.join(MISCLS_ROOT, f"{gt_name}_as_{pred_name}")
        os.makedirs(out_dir, exist_ok=True)

        img = Image.open(file_paths[idx]).convert("RGB")
        draw = ImageDraw.Draw(img)
        draw.rectangle([(0, 0), (img.width, 28)], fill=(0, 0, 0))
        draw.text((5, 5), f"GT:{gt_name} PRED:{pred_name}", fill=(255, 255, 0))
        img.save(os.path.join(out_dir, os.path.basename(file_paths[idx])))

print(f"\nMisclassified images saved in {MISCLS_ROOT}")
