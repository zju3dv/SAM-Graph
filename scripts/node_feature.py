import argparse, cv2, numpy as np, os
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--image_path", type=str, help="Path to the images")
parser.add_argument("--feature_path", type=str, help="Path to the SAM encoder features")
parser.add_argument("--sam_mask_path", type=str, help="Path to the SAM masks")
parser.add_argument("--node_feature_path", type=str, help="Path to the output node features")
parser.add_argument("--skip_if_exist", default=False, action='store_true', help="Whether to skip if target already exists")
args = parser.parse_args()

image_list = os.listdir(args.image_path)

h, w = cv2.imread(f'{args.image_path}/{image_list[0]}').shape[:2]
# Note: here we simply assume that all images are of the same resolution, which can be modified if needed.

feature_h, feature_w = 64, 64 # resolution of SAM encoder feature maps
if h < w:
    feature_h = int(feature_w * h / w)
else:
    feature_w = int(feature_h * w / h)

feature_dict = dict()

if args.skip_if_exist and os.path.exists(args.node_feature_path):
    print(f'{args.node_feature_path} already exists, skip.')
    exit()

for image_f in tqdm(image_list, desc='Calulating node features'):
    image_id = image_f.split('.')[0]
    mask_data = np.load(f'{args.sam_mask_path}/{image_id}.npz', allow_pickle=True)['arr_0'].tolist()
    points_per_instance, _, _ = mask_data['points_per_instance'], mask_data['masks'], mask_data['iou_preds']
    feature_map = np.load(f'{args.feature_path}/{image_id}.npy')
    for sp_id, pts in points_per_instance.items():
        xy = np.array(pts).astype(np.float32).T
        xy[0] = xy[0] / w * feature_w
        xy[1] = xy[1] / h * feature_h
        xy1 = xy.astype(np.int32)
        xy2 = xy1 + 1
        xy2[0] = xy2[0].clip(max=feature_w - 1)
        xy2[1] = xy2[1].clip(max=feature_h - 1)
        feature = feature_map[xy1[1], xy1[0]] * (xy2[0] - xy[0])[:, None] * (xy2[1] - xy[1])[:, None] + \
                    feature_map[xy2[1], xy1[0]] * (xy2[0] - xy[0])[:, None] * (xy[1] - xy1[1])[:, None] + \
                    feature_map[xy1[1], xy2[0]] * (xy[0] - xy1[0])[:, None] * (xy2[1] - xy[1])[:, None] + \
                    feature_map[xy2[1], xy2[0]] * (xy[0] - xy1[0])[:, None] * (xy[1] - xy1[1])[:, None]
        feature = feature.mean(0)
        if sp_id not in feature_dict:
            feature_dict[sp_id] = [feature]
        else:
            feature_dict[sp_id].append(feature)

feature_mean_dict = dict()
for k, v in feature_dict.items():
    feature_mean_dict[k] = sum(v) / len(v)

np.savez_compressed(args.node_feature_path, feature_mean_dict)
