import os, torch, torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from tqdm import tqdm
import open_clip
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from PIL import Image

# ---------------- CONFIG -------------------------------
DATA_DIR    = "/home/jayant/Desh4/Train"
BATCH_SIZE  = 1
EPOCHS      = 15
LR          = 2e-5
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID    = "hf-hub:microsoft/BioMedCLIP-PubMedBERT_256-vit_base_patch16_224"
NUM_CLASSES = 2
SAVE_DIR    = "/home/jayant/Desh4/finetune2"
os.makedirs(SAVE_DIR, exist_ok=True)

# ---------------- Albumentations Transform --------------
mean = [0.48145466, 0.4578275, 0.40821073]
std  = [0.26862954, 0.26130258, 0.27577711]

alb_transform = A.Compose([
    A.Resize(224, 224),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.4, hue=0.1, p=0.9),
    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.9),
    A.Normalize(mean=mean, std=std),
    ToTensorV2(),
])

# ---------------- Albumentations Wrapper -----------------
class AlbumentationsImageFolder(ImageFolder):
    def __init__(self, root, transform=None):
        super().__init__(root)
        self.alb_transform = transform

    def __getitem__(self, index):
        path, label = self.samples[index]
        img = self.loader(path).convert("RGB")
        img = self.alb_transform(image=np.array(img))["image"]
        return img, label

# ---------------- Model Definition -----------------------
class BioMedClipClassifier(nn.Module):
    def __init__(self, clip_model, num_classes=2):
        super().__init__()
        vit = clip_model.visual.trunk
        self.vit = vit
        self.embed_dim = vit.embed_dim
        vit.blocks = vit.blocks[:-1]

        # freeze all, unfreeze last 3 blocks
        for p in vit.parameters():
            p.requires_grad = False
        for blk in vit.blocks[-3:]:
            for p in blk.parameters():
                p.requires_grad = True

        self.bilinear = nn.Bilinear(self.embed_dim, self.embed_dim, self.embed_dim)
        self.classifier = nn.Linear(self.embed_dim, num_classes)

    def forward(self, x):
        x = self.vit.patch_embed(x)
        cls = self.vit.cls_token.expand(x.size(0), -1, -1)
        x = self.vit.pos_drop(torch.cat((cls, x), dim=1) + self.vit.pos_embed)

        cls_tokens = []
        for i, blk in enumerate(self.vit.blocks):
            x = blk(x)
            if i in [len(self.vit.blocks) - 3, len(self.vit.blocks) - 2]:
                cls_tokens.append(x[:, 0])

        pooled = self.bilinear(*cls_tokens)
        pooled = self.vit.norm(pooled)
        return self.classifier(pooled)

# ---------------- Load Data ------------------------------
dataset = AlbumentationsImageFolder(DATA_DIR, transform=alb_transform)
val_len = int(0.2 * len(dataset))
train_len = len(dataset) - val_len
train_set, val_set = random_split(dataset, [train_len, val_len])

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# ---------------- Setup Model + Optimizer ----------------
base_model, _ = open_clip.create_model_from_pretrained(MODEL_ID, device=DEVICE)
model = BioMedClipClassifier(base_model, NUM_CLASSES).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)

# ---------------- Train Loop -----------------------------
def run_epoch(loader, train=True):
    model.train() if train else model.eval()
    phase = "Train" if train else "Val"
    total_l, correct, total = 0.0, 0, 0

    with torch.set_grad_enabled(train):
        for imgs, labels in tqdm(loader, desc=f"{phase} batches", leave=False):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            if train: optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, labels)
            if train:
                loss.backward()
                optimizer.step()
            total_l += loss.item() * imgs.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            total += imgs.size(0)

    return total_l / total, correct / total

# ---------------- Epoch Loop -----------------------------
for epoch in range(1, EPOCHS + 1):
    print(f"Epoch {epoch}/{EPOCHS}")
    tr_loss, tr_acc = run_epoch(train_loader, train=True)
    vl_loss, vl_acc = run_epoch(val_loader,   train=False)
    print(f"  ➜ Train loss {tr_loss:.4f}  acc {tr_acc:.4f}")
    print(f"  ➜ Val   loss {vl_loss:.4f}  acc {vl_acc:.4f}")

    ckpt_path = os.path.join(SAVE_DIR, f"biomedclip_desh_finetune_{epoch}.pth")
    torch.save(model.state_dict(), ckpt_path)
    print(f"✅ Saved: {ckpt_path}")
