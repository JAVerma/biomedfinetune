import os, torch, torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from torchvision import transforms
from tqdm import tqdm
import open_clip

# --------------------- CONFIG -------------------------------
DATA_DIR    = "/home/jayant/Desh4/pasted_train"
BATCH_SIZE  = 4
EPOCHS      = 10
LR          = 2e-5
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID    = "hf-hub:microsoft/BioMedCLIP-PubMedBERT_256-vit_base_patch16_224"
NUM_CLASSES = 2
# ------------------------------------------------------------

# ---------- DATA -------------------------------------------
mean = [0.48145466, 0.4578275, 0.40821073]
std  = [0.26862954, 0.26130258, 0.27577711]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

dataset    = ImageFolder(DATA_DIR, transform=transform)
val_len    = int(0.2 * len(dataset))
train_len  = len(dataset) - val_len
train_set, val_set = random_split(dataset, [train_len, val_len])

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4)
val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# ---------- MODEL WRAPPER ----------------------------------
class BioMedClipClassifier(nn.Module):
    def __init__(self, clip_model, num_classes=2, n_train_blocks=3):
        super().__init__()
        vit            = clip_model.visual.trunk          # timm ViT
        self.vit       = vit
        self.embed_dim = vit.embed_dim
        vit.blocks     = vit.blocks[:-1]                  # drop last block

        # freeze, then unfreeze last n blocks
        for p in vit.parameters():      p.requires_grad = False
        for blk in vit.blocks[-n_train_blocks:]:
            for p in blk.parameters(): p.requires_grad = True

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

# ---------- LOAD BASE MODEL & OPTIMIZER --------------------
base_model, _ = open_clip.create_model_from_pretrained(MODEL_ID, device=DEVICE)
model         = BioMedClipClassifier(base_model, NUM_CLASSES).to(DEVICE)
criterion     = nn.CrossEntropyLoss()
optimizer     = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)

# ---------- TRAIN / VAL FUNCTIONS w/ tqdm ------------------
def run_epoch(loader, train=True):
    model.train() if train else model.eval()
    phase   = "Train" if train else "Val"
    total_l, correct, total = 0.0, 0, 0

    with torch.set_grad_enabled(train):
        for imgs, labels in tqdm(loader, desc=f"{phase} batches", leave=False):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

            if train:
                optimizer.zero_grad()
            logits = model(imgs)
            loss   = criterion(logits, labels)
            if train:
                loss.backward()
                optimizer.step()

            batch_size = imgs.size(0)
            total_l += loss.item() * batch_size
            correct += (logits.argmax(1) == labels).sum().item()
            total   += batch_size

    return total_l/total, correct/total

# ---------------- MAIN LOOP --------------------------------
for epoch in range(1, EPOCHS+1):
    print(f"\nEpoch {epoch}/{EPOCHS}")
    tr_loss, tr_acc = run_epoch(train_loader, train=True)
    vl_loss, vl_acc = run_epoch(val_loader,   train=False)
    print(f"  ➜ Train loss {tr_loss:.4f}  acc {tr_acc:.4f}")
    print(f"  ➜ Val   loss {vl_loss:.4f}  acc {vl_acc:.4f}")

# ---------------- SAVE -------------------------------------
    os.makedirs('./finetune', exist_ok=True)
    
    torch.save(model.state_dict(), os.path.join('./finetune', f"biomedclip_desh_finetune_{epoch}.pth"))
    print(f"\n✅ Model saved to biomedclip_desh_finetune_{epoch}.pth")
