import os
import subprocess

# Input paths
data_path = "../example_data/21d970d8de"
image_path = os.path.join(data_path, "images/")
colmap_path = os.path.join(data_path, "sparse/")
mesh_path = os.path.join(data_path, "rgb.ply")

# Dependencies
segmentor_path = "/home/guohaoyu/repos/ScanNet/Segmentator/segmentator"
sam_ckpt_path = "../sam_vit_h_4b8939.pth"

# Output paths
superpoint_path = os.path.join(data_path, "rgb.0.200000.segs.json")
feature_path = os.path.join(data_path, "sam_encoder_feature/")
superpoint_projection_path = os.path.join(data_path, "superpoint_projection/")
sam_mask_path = os.path.join(data_path, "sam_mask/")
graph_structure_path = os.path.join(data_path, "graph_structure.npz")
node_feature_path = os.path.join(data_path, "node_feature.npz")
edge_weights_path = os.path.join(data_path, "edge_weights.npz")
graph_segmentation_path = os.path.join(data_path, "graph_segmentation.npy")
graph_segmentation_visualize_path = os.path.join(data_path, "graph_segmentation_visualize.ply")

# Options
image_width = 640 # resize images to this width
skip_if_exist = 1
skip_arg = "--skip_if_exist" if skip_if_exist else ""

# Helper function to run shell commands
def run_command(command):
    print(f"Running: {command}")
    subprocess.run(command, shell=True, check=True)

# Commands
run_command(
    "python sam_encoder_feature.py "
    f"--image_path {image_path} "
    f"--image_width {image_width} "
    f"--feature_path {feature_path} "
    f"--sam_ckpt_path {sam_ckpt_path} "
    f"{skip_arg}"
)

if os.path.exists(superpoint_path) and skip_if_exist:
    print("Superpoint file already exists, skipping segmentor step")
else:
    run_command(f"{segmentor_path} {mesh_path} 0.2 50")

run_command(
    "python superpoint_projection.py "
    f"--image_path {image_path} "
    f"--image_width {image_width} "
    f"--colmap_path {colmap_path} "
    f"--mesh_path {mesh_path} "
    f"--superpoint_path {superpoint_path} "
    f"--superpoint_projection_path {superpoint_projection_path} "
    f"{skip_arg}"
)

run_command(
    "python predict_masks.py "
    f"--image_path {image_path} "
    f"--image_width {image_width} "
    f"--feature_path {feature_path} "
    f"--sam_ckpt_path {sam_ckpt_path} "
    f"--superpoint_projection_path {superpoint_projection_path} "
    f"--superpoint_path {superpoint_path} "
    f"--sam_mask_path {sam_mask_path} "
    f"{skip_arg}"
)

run_command(
    "python build_graph_structure.py "
    f"--mesh_path {mesh_path} "
    f"--superpoint_path {superpoint_path} "
    f"--graph_structure_path {graph_structure_path} "
    f"{skip_arg}"
)

run_command(
    "python node_feature.py "
    f"--image_path {image_path} "
    f"--feature_path {feature_path} "
    f"--sam_mask_path {sam_mask_path} "
    f"--node_feature_path {node_feature_path} "
    f"{skip_arg}"
)

run_command(
    "python edge_weights.py "
    f"--image_path {image_path} "
    f"--graph_structure_path {graph_structure_path} "
    f"--sam_mask_path {sam_mask_path} "
    f"--edge_weights_path {edge_weights_path} "
    f"{skip_arg}"
)

# TODO
# run_command(
#     "python graph_segmentation.py "
#     f"--image_path {image_path} "
#     f"--mesh_path {mesh_path} "
#     f"--superpoint_path {superpoint_path} "
#     f"--graph_structure_path {graph_structure_path} "
#     f"--sam_mask_path {sam_mask_path} "
#     f"--edge_weights_path {edge_weights_path} "
#     f"--graph_segmentation_path {graph_segmentation_path} "
#     f"--graph_segmentation_visualize_path {graph_segmentation_visualize_path} "
#     f"{skip_arg}"
# )
