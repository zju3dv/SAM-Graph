import argparse, trimesh, numpy as np, json, os
from tqdm import tqdm, trange
from scipy.spatial import KDTree

parser = argparse.ArgumentParser()
parser.add_argument("--mesh_path", type=str, help="Path to the mesh (or point cloud)")
parser.add_argument("--superpoint_path", type=str, help="Path to the superpoint segmentation (.json)")
parser.add_argument("--graph_structure_path", type=str, help="Path to the output graph structure (.npz)")
parser.add_argument("--skip_if_exist", default=False, action='store_true', help="Whether to skip if target already exists")
args = parser.parse_args()

if args.skip_if_exist and os.path.exists(args.graph_structure_path):
    print(f'{args.graph_structure_path} already exists, skip.')
    exit()

m = trimesh.load(args.mesh_path)
seg = json.load(open(args.superpoint_path))
seg_indices = np.array(seg['segIndices'])
sp_ids = np.unique(seg_indices)
mapping = np.zeros((sp_ids.max() + 1, ), dtype=np.int32)
for i, sp_id in enumerate(sp_ids):
    mapping[sp_id] = i
seg_indices = mapping[seg_indices]

pc_list = []
for sp_id in np.unique(seg_indices):
    pc = m.vertices[seg_indices == sp_id]
    pc = np.asarray(pc)
    if pc.shape[0] > 50:
        pc = pc[np.random.choice(pc.shape[0], size=50, replace=False)]
    pc_list.append(pc)

distances = {}
for i in trange(len(pc_list) - 1, desc='Building graph structure'):
    tree_i = KDTree(pc_list[i])
    pc_concat = np.concatenate(pc_list[i + 1:])
    dist_concat = tree_i.query(pc_concat)[0]
    dist_split = np.split(dist_concat, np.cumsum([len(pc) for pc in pc_list[i+1:]]))[:-1]
    for j, dist in enumerate(dist_split):
        distances[(i, i + j + 1)] = dist.min()
    # for j in range(i + 1, len(pc_list)):
    #     dist, _ = tree_i.query(pc_list[j])
    #     dist = min(dist)
    #     distances[(i, j)] = dist

edges = []
for (a, b), distance in distances.items():
    if distance < 0.3:
        edges.append((a, b))

np.savez_compressed(args.graph_structure_path, np.asarray(edges))
