import torch
from PIL import Image
from transformers import AutoTokenizer, AutoModel
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode


MODEL_ID   = "dkhursen/InternVL2-2b-LoRA-25k_plus_DL-PL-10pct"
IMAGE_PATH = "nuscenes/stitched/f9e460f092c94466b1211704b5a8859d_33e36dbd62594a10b783b710350b100f.jpg"   # Replace with your image
# Question with image token
QUESTION   = "<image>\nWhat are the important objects in the current scene? Those objects will be considered for the future reasoning and driving decision."
    

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


def build_transform(input_size):
    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    """Pick the tile grid whose aspect ratio best matches the image."""
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        ratio_diff = abs(aspect_ratio - ratio[0] / ratio[1])
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            # prefer larger grids for high-res images
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    """
    Split image into tiles that match its native aspect ratio.
    Optionally appends a downsampled full-image thumbnail as the last tile.
    """
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = sorted(
        {(i, j)
         for n in range(min_num, max_num + 1)
         for i in range(1, n + 1)
         for j in range(1, n + 1)
         if min_num <= i * j <= max_num},
        key=lambda x: x[0] * x[1],
    )

    best_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )
    target_w = image_size * best_ratio[0]
    target_h = image_size * best_ratio[1]
    cols     = best_ratio[0]

    resized = image.resize((target_w, target_h))
    tiles   = []
    for i in range(best_ratio[0] * best_ratio[1]):
        col = i % cols
        row = i // cols
        box = (col * image_size, row * image_size,
               (col + 1) * image_size, (row + 1) * image_size)
        tiles.append(resized.crop(box))

    if use_thumbnail and len(tiles) != 1:
        tiles.append(image.resize((image_size, image_size)))

    return tiles


def load_image(image_path, input_size=448, max_num=12):
    image     = Image.open(image_path).convert("RGB")
    transform = build_transform(input_size)
    tiles     = dynamic_preprocess(image, image_size=input_size,
                                   use_thumbnail=True, max_num=max_num)
    return torch.stack([transform(tile) for tile in tiles])



# Load tokenizer and model from Hugging Face Hub
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True, use_fast=False)
model = AutoModel.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
).eval().cuda()

# Preprocess image into tiles expected by InternVL2
pixel_values = load_image(IMAGE_PATH, max_num=12).to(torch.bfloat16).cuda()

generation_config = {"max_new_tokens": 512, "do_sample": False}

print("Question:", QUESTION)
response = model.chat(tokenizer, pixel_values, QUESTION, generation_config)
print("Answer:", response)

