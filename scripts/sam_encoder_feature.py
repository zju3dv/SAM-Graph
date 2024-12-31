import argparse, cv2, numpy as np, os
from tqdm import tqdm
from segment_anything import build_sam, SamPredictor

parser = argparse.ArgumentParser()
parser.add_argument("--image_path", type=str, help="Path to the images")
parser.add_argument("--image_width", type=float, default=640, help="Width of the resized images")
parser.add_argument("--feature_path", type=str, help="Path to the output SAM encoder features")
parser.add_argument("--sam_ckpt_path", type=str, help="Path to the SAM model parameters")
parser.add_argument("--skip_if_exist", default=False, action='store_true', help="Whether to skip if target already exists")
args = parser.parse_args()

sam = None

# sam = build_sam(checkpoint=args.sam_ckpt_path)
# sam.to(device='cuda')
# sam_predictor = SamPredictor(sam)

image_list = os.listdir(args.image_path)

os.makedirs(args.feature_path, exist_ok=True)

for image_f in tqdm(image_list, desc='Extracting SAM encoder features'):
    image_id = image_f.split('.')[0]
    if os.path.exists(f'{args.feature_path}/{image_id}.npy') and args.skip_if_exist:
        continue
    if sam is None:
        sam = build_sam(checkpoint=args.sam_ckpt_path)
        sam.to(device='cuda')
        sam_predictor = SamPredictor(sam)
    img = cv2.imread(f'{args.image_path}/{image_f}')
    height_original, width_original = img.shape[:2]
    w = int(args.image_width)
    h = int(w * height_original / width_original)
    img = cv2.resize(img, (w, h))
    sam_predictor.set_image(img)
    feature_map = sam_predictor.features[0].permute(1, 2, 0).cpu().numpy()
    np.save(f'{args.feature_path}/{image_id}.npy', feature_map)
