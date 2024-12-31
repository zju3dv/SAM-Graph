import argparse, cv2, trimesh, numpy as np, os, pyrender
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--image_path", type=str, help="Path to the images")
parser.add_argument("--pose_path", type=str, help="Path to the camera poses (camera to world)")
parser.add_argument("--intrinsic_path", type=str, help="Path to the intrinsic matrix (.txt)")
parser.add_argument("--mesh_path", type=str, help="Path to the mesh (or point cloud)")
parser.add_argument("--rendered_depth_path", type=str, help="Path to the rendered depth maps")
parser.add_argument("--rendered_depth_vis_path", type=str, default='', help="Path to the visualization of rendered depth maps (optional)")
args = parser.parse_args()

class Renderer():
    def __init__(self, height=1440, width=1440):
        self.renderer = pyrender.OffscreenRenderer(width, height)
        self.scene = pyrender.Scene()
        # self.render_flags = pyrender.RenderFlags.SKIP_CULL_FACES

    def __call__(self, height, width, intrinsics, pose, mesh):
        self.renderer.viewport_height = height
        self.renderer.viewport_width = width
        self.scene.clear()
        self.scene.add(mesh)
        cam = pyrender.IntrinsicsCamera(cx=intrinsics[0, 2], cy=intrinsics[1, 2],
                                        fx=intrinsics[0, 0], fy=intrinsics[1, 1])
        self.scene.add(cam, pose=self.fix_pose(pose))
        return self.renderer.render(self.scene)  # , self.render_flags)

    def fix_pose(self, pose):
        # 3D Rotation about the x-axis.
        t = np.pi
        c = np.cos(t)
        s = np.sin(t)
        R = np.array([[1, 0, 0],
                      [0, c, -s],
                      [0, s, c]])
        axis_transform = np.eye(4)
        axis_transform[:3, :3] = R
        return pose @ axis_transform

    def mesh_opengl(self, mesh):
        material = pyrender.MetallicRoughnessMaterial(
            baseColorFactor=[1, 1, 1, 1.],
            metallicFactor=0.0,
            roughnessFactor=0.0,
            smooth=False,
            alphaMode='OPAQUE')
        return pyrender.Mesh.from_trimesh(mesh, material=material)

    def delete(self):
        self.renderer.delete()

image_list = os.listdir(args.image_path)

h, w = cv2.imread(f'{args.image_path}/{image_list[0]}').shape[:2]
# Note: here we simply assume that all images are of the same resolution

os.makedirs(args.superpoint_projection_path, exist_ok=True)

ixt = np.loadtxt(args.intrinsic_path)

c2ws = dict()
for image_f in image_list:
    image_id = image_f.split('.')[0]
    c2w = np.loadtxt(f'{args.pose_path}/{image_id}.txt')
    c2ws[image_id] = c2w

m = trimesh.load(args.mesh_path)

renderer = Renderer()
mesh_opengl = renderer.mesh_opengl(m)

os.makedirs(args.rendered_depth_path, exist_ok=True)
if args.rendered_depth_vis_path != '':
    os.makedirs(args.rendered_depth_vis_path, exist_ok=True)

for k, c2w in tqdm(c2ws.items()):
    _, depth = renderer(h, w, ixt, c2w, mesh_opengl)
    np.savez_compressed(f'{args.rendered_depth_path}/{k}.npz', depth)
    
    if args.rendered_depth_vis_path != '':
        depth = (depth - depth.min()) / (depth.max() - depth.min())
        depth = (depth * 255).astype(np.uint8)
        depth_vis = cv2.applyColorMap(depth, cv2.COLORMAP_JET)
        cv2.imwrite(f'{args.rendered_depth_vis_path}/{k}.jpg', depth_vis)
