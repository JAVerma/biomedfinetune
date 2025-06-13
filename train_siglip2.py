import os, torch, torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from tqdm import tqdm
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from transformers import SiglipProcessor, SiglipModel

# ---------------- CONFIG -------------------------------
DATA_DIR    = "/home/jayant/Desh4/pasted_train2"
BATCH_SIZE  = 4
EPOCHS      = 15
LR          = 2e-5
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID    = "google/siglip2-base-patch16-384"
NUM_CLASSES = 2
SAVE_DIR    = "/home/jayant/Desh4/finetune_siglip"
os.makedirs(SAVE_DIR, exist_ok=True)

# ---------------- Processor ----------------------------
processor = SiglipProcessor.from_pretrained(MODEL_ID)
image_mean = processor.image_processor.image_mean
image_std  = processor.image_processor.image_std
image_size = processor.image_processor.size["height"]

# ---------------- Albumentations Transform --------------
alb_transform = A.Compose([
    A.Resize(image_size, image_size),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.4, hue=0.1, p=0.9),
    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.9),
    A.Normalize(mean=image_mean, std=image_std),
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
class SigLIPClassifier(nn.Module):
    def __init__(self, model: SiglipModel, num_classes=2, n_train_blocks=3):
        super().__init__()
        self.vision_model = model.vision_model
        self.hidden_size = self.vision_model.config.hidden_size
        self.transformer = self.vision_model.encoder

        # Drop final transformer block
        self.transformer.layers = self.transformer.layers[:-1]

        # Freeze all parameters
        for param in self.vision_model.parameters():
            param.requires_grad = False

        # Unfreeze last n transformer blocks
        for blk in self.transformer.layers[-n_train_blocks:]:
            for param in blk.parameters():
                param.requires_grad = True

        self.bilinear   = nn.Bilinear(self.hidden_size, self.hidden_size, self.hidden_size)
        self.classifier = nn.Linear(self.hidden_size, num_classes)

    def forward(self, pixel_values):
        # Step 1: get embeddings
        x = self.vision_model.embeddings(pixel_values)

        # Step 2: attention mask: [B, S] → [B, 1, 1, S]
        batch_size, seq_len, _ = x.shape
        attention_mask = torch.ones((batch_size, seq_len), dtype=torch.bool, device=pixel_values.device)
        attention_mask = attention_mask[:, None, None, :]  # shape: [B, 1, 1, S]

        cls_tokens = []
        for i, blk in enumerate(self.transformer.layers):
            x = blk(x, attention_mask=attention_mask)[0]
            if i in [len(self.transformer.layers) - 3, len(self.transformer.layers) - 2]:
                cls_tokens.append(x[:, 0])  # CLS token

        pooled = self.bilinear(*cls_tokens)
        pooled = self.vision_model.post_layernorm(pooled)
        return self.classifier(pooled)

# ---------------- Load Data ------------------------------
dataset = AlbumentationsImageFolder(DATA_DIR, transform=alb_transform)
val_len = int(0.2 * len(dataset))
train_len = len(dataset) - val_len
train_set, val_set = random_split(dataset, [train_len, val_len])

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# ---------------- Load Base Model ------------------------
base_model = SiglipModel.from_pretrained(MODEL_ID).to(DEVICE)
model = SigLIPClassifier(base_model, NUM_CLASSES).to(DEVICE)
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
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            if train:
                loss.backward()
                optimizer.step()

            total_l += loss.item() * imgs.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += imgs.size(0)

    return total_l / total, correct / total

# ---------------- Epoch Loop -----------------------------
for epoch in range(1, EPOCHS + 1):
    print(f"\n📦 Epoch {epoch}/{EPOCHS}")
    tr_loss, tr_acc = run_epoch(train_loader, train=True)
    vl_loss, vl_acc = run_epoch(val_loader,   train=False)
    print(f"  ✅ Train loss {tr_loss:.4f}  acc {tr_acc:.4f}")
    print(f"  📊 Val   loss {vl_loss:.4f}  acc {vl_acc:.4f}")

    ckpt_path = os.path.join(SAVE_DIR, f"siglip_desh_finetune_{epoch}.pth")
    torch.save(model.state_dict(), ckpt_path)
    print(f"💾 Saved: {ckpt_path}")
