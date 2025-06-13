import os, glob
from PIL import Image, ImageOps

# Collect image paths
images = glob.glob('./Train/*/*')

for img_path in images:
    # Load original image and corresponding mask
    img = Image.open(img_path).convert("RGB")
    mask_path = img_path.replace('Train', 'Train_mask')
    mask_path=mask_path.replace('.png','_mask.png')
    mask = Image.open(mask_path).convert("L")  # ensure mask is single-channel (grayscale)

    # Invert the mask
    inverted_mask = ImageOps.invert(mask)

    # Create a red overlay (same size as image)
    red_overlay = Image.new("RGB", img.size, color=(255, 0, 0))

    # Paste red onto original image using inverted mask as transparency
    img_with_overlay = Image.composite(red_overlay, img, inverted_mask)

    # Show or save result
    dir_name=os.path.dirname(img_path).split('/')[-1]
    os.makedirs(os.path.join('./pasted_train/',dir_name.split('/')[-1]),exist_ok=True)
    print(os.path.join('./pasted_train/',dir_name.split('/')[-1]))
    img_with_overlay.save(f'./pasted_train/{dir_name}/{os.path.basename(img_path)}')  # or replace with img_with_overlay.save("output_path.png")
    # break
