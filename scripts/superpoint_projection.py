import argparse, cv2, trimesh, numpy as np, os, json, torch, pycolmap
from tqdm import tqdm
from pytorch3d.renderer import RasterizationSettings, MeshRasterizer, FoVPerspectiveCameras
from pytorch3d.structures import Meshes
from PIL import Image

def getProjectionMatrixK(znear, zfar, K, H, W):
    P = torch.zeros(4, 4)
    z_sign = 1.0

    fx = K[..., 0, 0]
    fy = K[..., 1, 1]
    cx = K[..., 0, 2]
    cy = K[..., 1, 2]
    s = K[..., 0, 1]

    P[0, 0] = 2 * fx / H
    P[1, 1] = 2 * fy / H
    P[0, 1] = 2 * s / W
    
    P[2, 2] = z_sign * (zfar + znear) / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    P[3, 2] = z_sign
    
    P[0, 2] = 1 - 2 * (cx / W)
    P[1, 2] = 1 - 2 * (cy / H)
    
    return P

parser = argparse.ArgumentParser()
parser.add_argument("--image_path", type=str, help="Path to the images")
parser.add_argument("--image_width", type=float, default=640, help="Width of the resized images")
parser.add_argument("--colmap_path", type=str, help="Path to the colmap SfM results (camera poses and intrinsics)")
parser.add_argument("--mesh_path", type=str, help="Path to the mesh (or point cloud)")
parser.add_argument("--superpoint_path", type=str, help="Path to the superpoint segmentation (.json)")
parser.add_argument("--superpoint_projection_path", type=str, help="Path to the output superpoint projection masks")
parser.add_argument("--skip_if_exist", default=False, action='store_true', help="Whether to skip if target already exists")
args = parser.parse_args()

image_list = os.listdir(args.image_path)

m = None

os.makedirs(args.superpoint_projection_path, exist_ok=True)

sfm_results = pycolmap.Reconstruction(args.colmap_path)

for image in tqdm(sfm_results.images.values(), desc='Superpoint projection'):
    w2c = np.eye(4)
    w2c[:3] = image.cam_from_world.matrix()
    c2w = np.linalg.inv(w2c)
    fx, fy, cx, cy = image.camera.params[:4]
    ixt = np.array([[fx, 0, cx, 0], [0, fy, cy, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    height_original, width_original = image.camera.height, image.camera.width
    w = int(args.image_width)
    h = int(args.image_width * height_original / width_original)
    ixt[:2] *= w / width_original

    if os.path.exists(f'{args.superpoint_projection_path}/{image.name.split(".")[0]}.png') and args.skip_if_exist:
        continue

    if m is None:
        m = trimesh.load(args.mesh_path)
        seg = json.load(open(args.superpoint_path))
        seg_indices = np.array(seg['segIndices'])

        mesh = Meshes(verts=[torch.from_numpy(m.vertices).float()], faces=[torch.from_numpy(m.faces)]).cuda()
        faces = torch.tensor(m.faces)

        num_seg = seg_indices.max() + 1
        random_colors = torch.zeros((num_seg, 3), dtype=torch.uint8)
        for i in np.unique(seg_indices):
            random_colors[i][0] = (i >> 16) & 255
            random_colors[i][1] = (i >> 8) & 255
            random_colors[i][2] = i & 255

    K = getProjectionMatrixK(0.0001, 100, ixt, h, w).numpy()
    
    R, t = c2w[:3, :3].copy(), c2w[:3, 3].copy()
    R[:, 0] = -R[:, 0]
    R[:, 1] = -R[:, 1]
    c2w[:3, :3] = R
    c2w[:3, 3] = t
    w2c = np.linalg.inv(c2w.copy())
    R, t = w2c[:3, :3].copy(), w2c[:3, 3].copy()
    R = R.T
    
    cameras = FoVPerspectiveCameras(device=torch.device('cuda'), R=R[None].astype(np.float32), T=t[None].astype(np.float32), K=K[None].astype(np.float32))
    raster_settings = RasterizationSettings(image_size=(h, w))
    rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)
    fragments = rasterizer(mesh)

    face_ids_per_pixel = fragments.pix_to_face[..., 0].cpu()
    
    vertice_id = faces[face_ids_per_pixel[0]]
    seg_id = torch.from_numpy(seg_indices)[vertice_id]
    consistent_mask = (seg_id[..., 0] == seg_id[..., 1]) & (seg_id[..., 1] == seg_id[..., 2])
    edge_mask = ~consistent_mask
    image_colors = random_colors[seg_id[..., 0]]
    background_color = torch.tensor([255, 255, 255], dtype=torch.uint8)
    mask = (face_ids_per_pixel[0] == -1) | edge_mask
    image_colors[mask] = background_color
    
    overseg_vis = image_colors.cpu().numpy()

    Image.fromarray(overseg_vis).save(f'{args.superpoint_projection_path}/{image.name.split(".")[0]}.png') 
