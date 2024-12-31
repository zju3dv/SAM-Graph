import argparse, numpy as np, os
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--image_path", type=str, help="Path to the images")
parser.add_argument("--graph_structure_path", type=str, help="Path to the graph structure (.npz)")
parser.add_argument("--sam_mask_path", type=str, help="Path to the SAM masks")
parser.add_argument("--edge_weights_path", type=str, help="Path to the output edge weights (.npz)")
parser.add_argument("--skip_if_exist", default=False, action='store_true', help="Whether to skip if target already exists")
args = parser.parse_args()

if args.skip_if_exist and os.path.exists(args.edge_weights_path):
    print(f'{args.edge_weights_path} already exists, skip.')
    exit()

def select_mask(masks, iou_preds):
    if iou_preds[2] > iou_preds.max() - 0.05:
        return masks[2], iou_preds[2]
    elif iou_preds[1] > iou_preds[0] - 0.05:
        return masks[1], iou_preds[1]
    else:
        return masks[0], iou_preds[0]

def weighted_average(data):
    score_sum = 0
    distance_sum = 0
    for k, (score, score1, score2, iou_pred1, iou_pred2, distance) in data.items():
        score_sum += max(score1, score2) * distance * iou_pred1 * iou_pred2
        distance_sum += distance * iou_pred1 * iou_pred2
    return score_sum / distance_sum

edges = np.load(args.graph_structure_path)['arr_0']

image_list = os.listdir(args.image_path)

edge_weights = dict()

for image_f in tqdm(image_list, desc='Calculating edge weights'):
    image_id = image_f.split('.')[0]
    mask_data = np.load(f'{args.sam_mask_path}/{image_id}.npz', allow_pickle=True)['arr_0'].tolist()
    points_per_instance, masks, iou_preds = mask_data['points_per_instance'], mask_data['masks'], mask_data['iou_preds']

    for p1, p2 in edges:
        if p1 not in points_per_instance.keys():
            continue
        if p2 not in points_per_instance.keys():
            continue
        if (p1, p2) not in edge_weights:
            edge_weights[(p1, p2)] = dict()
            assert (p2, p1) not in edge_weights
            edge_weights[(p2, p1)] = dict()
        xy1 = points_per_instance[p1]
        xy2 = points_per_instance[p2]
        
        distance2d = ((np.array(xy1).mean(0) - np.array(xy2).mean(0)) ** 2).sum() ** 0.5
        p1_id, p2_id = list(points_per_instance.keys()).index(p1), list(points_per_instance.keys()).index(p2)
        
        mask1, mask2 = masks[p1_id], masks[p2_id]
        iou_pred1, iou_pred2 = iou_preds[p1_id], iou_preds[p2_id]
        
        mask1, iou_pred1 = select_mask(mask1, iou_pred1)
        mask2, iou_pred2 = select_mask(mask2, iou_pred2)

        iou = (mask1 & mask2).sum() / (mask1 | mask2).sum()
        ioa = (mask1 & mask2).sum() / mask1.sum()
        iob = (mask1 & mask2).sum() / mask2.sum()
        
        edge_weights[(p1, p2)][image_id] = [iou, ioa, iob, iou_pred1, iou_pred2, distance2d]
        edge_weights[(p2, p1)][image_id] = [iou, iob, ioa, iou_pred2, iou_pred1, distance2d]

for k, v in edge_weights.items():
    edge_weights[k] = weighted_average(v)

np.savez(args.edge_weights_path, edge_weights)
