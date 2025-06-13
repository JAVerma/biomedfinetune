import os
import torch
import open_clip
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from multiprocessing import Process, Queue, set_start_method

# === CONFIG ===
label0_gt = "/home/jayant/Desh4/pasted_train/Desh"
label1_gt = "/home/jayant/Desh4/pasted_train/non-Desh"

text_classes = [
    "Normal brain MRI",
    "Brain MRI showing Disproportionately Enlarged Subarachnoid Space Hydrocephalus (DESH)",
]

siglip_model_name = "google/siglip2-base-patch16-384"
openclip_model_name = "ViT-H-14"
biomedclip_model_id = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_images(label0_dir, label1_dir):
    paths = []
    labels = []
    for fname in sorted(os.listdir(label0_dir)):
        if fname.endswith(('.jpg', '.png', '.jpeg')):
            paths.append(os.path.join(label0_dir, fname))
            labels.append(text_classes[1])
    for fname in sorted(os.listdir(label1_dir)):
        if fname.endswith(('.jpg', '.png', '.jpeg')):
            paths.append(os.path.join(label1_dir, fname))
            labels.append(text_classes[0])
    return paths, labels


def run_siglip2(image_paths, labels, result_queue):
    from transformers import SiglipProcessor, SiglipModel

    processor = SiglipProcessor.from_pretrained(siglip_model_name)
    model = SiglipModel.from_pretrained(siglip_model_name).to(device).eval()

    text_inputs = processor(text=text_classes, padding=True, return_tensors="pt").to(device)
    with torch.no_grad():
        text_embeds = model.get_text_features(**text_inputs)
        text_embeds /= text_embeds.norm(dim=-1, keepdim=True)

    preds = []
    for path in tqdm(image_paths, desc="SigLIP2"):
        img = Image.open(path).convert("RGB")
        inputs = processor(images=img, return_tensors="pt").to(device)
        with torch.no_grad():
            img_emb = model.get_image_features(**inputs)
            img_emb /= img_emb.norm(dim=-1, keepdim=True)
        sim = img_emb @ text_embeds.T
        preds.append(text_classes[sim.argmax().item()])

    acc = accuracy_score(labels, preds)
    result_queue.put((siglip_model_name, acc))


def run_openclip(image_paths, labels, result_queue):
    model, _, preprocess = open_clip.create_model_and_transforms(
        openclip_model_name, pretrained="laion2b_s32b_b79k", device=device
    )
    tokenizer = open_clip.get_tokenizer(openclip_model_name)

    with torch.no_grad():
        tokens = tokenizer(text_classes).to(device)
        text_embeds = model.encode_text(tokens)
        text_embeds /= text_embeds.norm(dim=-1, keepdim=True)

    preds = []
    for path in tqdm(image_paths, desc="OpenCLIP"):
        img = Image.open(path).convert("RGB")
        img_tensor = preprocess(img).unsqueeze(0).to(device)
        with torch.no_grad():
            img_emb = model.encode_image(img_tensor)
            img_emb /= img_emb.norm(dim=-1, keepdim=True)
        sim = img_emb @ text_embeds.T
        preds.append(text_classes[sim.argmax().item()])

    acc = accuracy_score(labels, preds)
    result_queue.put((f"openclip/{openclip_model_name}", acc))


def run_biomedclip(image_paths, labels, result_queue):
    model, preprocess = open_clip.create_model_from_pretrained(biomedclip_model_id, device=device)
    tokenizer = open_clip.get_tokenizer(biomedclip_model_id)

    with torch.no_grad():
        text_tokens = tokenizer(text_classes).to(device)
        text_embeds = model.encode_text(text_tokens)
        text_embeds /= text_embeds.norm(dim=-1, keepdim=True)

    preds = []
    for path in tqdm(image_paths, desc="BioMedCLIP"):
        img = Image.open(path).convert("RGB")
        img_tensor = preprocess(img).unsqueeze(0).to(device)
        with torch.no_grad():
            img_emb = model.encode_image(img_tensor)
            img_emb /= img_emb.norm(dim=-1, keepdim=True)
        sim = img_emb @ text_embeds.T
        preds.append(text_classes[sim.argmax().item()])

    acc = accuracy_score(labels, preds)
    result_queue.put(("microsoft/BioMedCLIP", acc))


if __name__ == "__main__":
    set_start_method("spawn", force=True)
    result_queue = Queue()

    image_paths, ground_truth = load_images(label0_gt, label1_gt)

    processes = [
        Process(target=run_siglip2, args=(image_paths, ground_truth, result_queue)),
        Process(target=run_openclip, args=(image_paths, ground_truth, result_queue)),
        Process(target=run_biomedclip, args=(image_paths, ground_truth, result_queue)),
    ]

    for p in processes:
        p.start()

    for p in processes:
        p.join()

    results = {}
    while not result_queue.empty():
        model_name, acc = result_queue.get()
        results[model_name] = acc

    print("\n✅ Results Summary:")
    for model, acc in results.items():
        print(f"{model} → Accuracy: {acc:.4f}")

    best_model = max(results.items(), key=lambda x: x[1])[0]
    print(f"\n🏆 Best model to fine-tune: **{best_model}**")
