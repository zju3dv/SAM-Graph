import argparse, cv2, numpy as np, os, json, torch
np.set_printoptions(suppress=True)
from PIL import Image
from tqdm import tqdm
from segment_anything import build_sam, SamPredictor


def remove_small_masks_from_segmentation(segmentation_mask, min_size=100):
    unique_ids = np.unique(segmentation_mask)
    for instance_id in unique_ids:
        if instance_id == -1:
            continue

        instance_mask = (segmentation_mask == instance_id).astype(np.uint8)
        
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(instance_mask, connectivity=8)
        
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] < min_size:
                segmentation_mask[labels == i] = -1
    return segmentation_mask


def sample_points_from_mask(mask, num_points=10):
    h, w = mask.shape
    
    extended_mask = np.zeros((h+2, w+2), dtype=np.uint8)
    extended_mask[1:-1, 1:-1] = mask

    distance_transform = cv2.distanceTransform(extended_mask, cv2.DIST_L2, 5)
    
    distance_transform = distance_transform[1:-1, 1:-1]
    sampled_points = []

    for _ in range(num_points):
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(distance_transform)
        sampled_points.append(max_loc)
        
        cv2.circle(distance_transform, max_loc, int(max_val), 0, -1)

    return sampled_points


def sample_points_for_each_instance(segmentation_mask, num_points=10):
    ret = dict()
    unique_ids = np.unique(segmentation_mask)

    for instance_id in unique_ids:
        if instance_id == -1:
            continue
        
        instance_mask = (segmentation_mask == instance_id).astype(np.uint8) * 255
        
        points = sample_points_from_mask(instance_mask, num_points=num_points)
        ret[instance_id] = points

    return ret


parser = argparse.ArgumentParser()
parser.add_argument("--image_path", type=str, help="Path to the images")
parser.add_argument("--image_width", type=float, default=640, help="Width of the resized images")
parser.add_argument("--sam_ckpt_path", type=str, help="Path to the SAM model parameters")
parser.add_argument("--feature_path", type=str, help="Path to the SAM encoder features")
parser.add_argument("--superpoint_projection_path", type=str, help="Path to the superpoint projection masks")
# parser.add_argument("--depth_difference_path", type=str, help="Path to the depth difference")
parser.add_argument("--superpoint_path", type=str, help="Path to the superpoint segmentation (.json)")
parser.add_argument("--sam_mask_path", type=str, help="Path to the SAM masks")
parser.add_argument("--skip_if_exist", default=False, action='store_true', help="Whether to skip if target already exists")
args = parser.parse_args()

sam = None

os.makedirs(args.sam_mask_path, exist_ok=True)

image_list = os.listdir(args.image_path)

height_original, width_original = cv2.imread(f'{args.image_path}/{image_list[0]}').shape[:2]
w = int(args.image_width)
h = int(w * height_original / width_original)
# Note: here we simply assume that all images are of the same resolution, which can be modified if needed.

for image_f in tqdm(image_list, desc='Predicting masks'):
    image_id = image_f.split('.')[0]

    if args.skip_if_exist and os.path.exists(f'{args.sam_mask_path}/{image_id}.npz'):
        continue

    if sam is None:
        sam = build_sam(checkpoint=args.sam_ckpt_path)
        sam.to(device='cuda')
        sam_predictor = SamPredictor(sam)

        seg = json.load(open(args.superpoint_path))
        seg_indices = np.array(seg['segIndices'])
        sp_ids = np.unique(seg_indices)
        mapping = np.zeros((sp_ids.max() + 1, ), dtype=np.int32)
        for i, sp_id in enumerate(sp_ids):
            mapping[sp_id] = i
        seg_indices = mapping[seg_indices]

    if not sam_predictor.is_image_set:
        img = cv2.imread(f'{args.image_path}/{image_f}')
        img = cv2.resize(img, (w, h))
        sam_predictor.set_image(img)
    else:
        sam_predictor.features = torch.from_numpy(np.load(f'{args.feature_path}/{image_id}.npy')).permute(2, 0, 1)[None].cuda()

    view_overseg = Image.open(f'{args.superpoint_projection_path}/{image_id}.png')
    view_overseg = np.asarray(view_overseg).astype(np.uint32)
    view_overseg = (view_overseg[..., 0] << 16) | (view_overseg[..., 1] << 8) | view_overseg[..., 2]
    bg = (view_overseg == 256 ** 3 - 1)
    # mask = Image.open(f'{args.rendered_depth_path}/{image_id}.png')
    # mask = np.array(mask) > 127
    # bg = bg | (~mask)
    view_overseg[bg] = 0
    view_overseg = mapping[view_overseg]
    view_overseg[bg] = -1
    view_overseg = remove_small_masks_from_segmentation(view_overseg)

    points_per_instance = sample_points_for_each_instance(view_overseg, num_points=5)
    points = np.array([list(_) for _ in points_per_instance.values()])

    transformed_points = sam_predictor.transform.apply_coords(points, (h, w))
    in_points = torch.as_tensor(transformed_points, device=sam_predictor.device)
    in_labels = torch.ones((in_points.shape[0], in_points.shape[1]), dtype=torch.int, device=in_points.device)
    
    masks_logits, iou_preds, _ = sam_predictor.predict_torch(
        in_points,
        in_labels,
        multimask_output=True,
        return_logits=True,
    )
    
    masks_logits = masks_logits.cpu().numpy()
    masks = masks_logits > 0
    iou_preds = iou_preds.cpu().numpy()

    data = {'points_per_instance': points_per_instance, 'masks': masks, 'iou_preds': iou_preds}
    np.savez_compressed(f'{args.sam_mask_path}/{image_id}.npz', data)
