from pathlib import Path
import open3d as o3d


OBJ_PATH = "/home/max/Downloads/model_tf.obj"
PLY_PATH = "/home/max/Downloads/scene_complete.ply"


def main():
    obj_path = Path(OBJ_PATH)
    ply_path = Path(PLY_PATH)

    if not obj_path.exists():
        raise FileNotFoundError(f"OBJ file not found: {obj_path}")
    if not ply_path.exists():
        raise FileNotFoundError(f"PLY file not found: {ply_path}")

    mesh = o3d.io.read_triangle_mesh(str(obj_path))
    if mesh.is_empty():
        raise ValueError(f"Loaded empty mesh from: {obj_path}")
    mesh.compute_vertex_normals()

    point_cloud = o3d.io.read_point_cloud(str(ply_path))
    if point_cloud.is_empty():
        raise ValueError(f"Loaded empty point cloud from: {ply_path}")

    if not point_cloud.has_colors():
        point_cloud.paint_uniform_color([0.1, 0.7, 1.0])

    coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)

    o3d.visualization.draw_geometries(
        [mesh, point_cloud, coord],
        window_name="OBJ + PLY Viewer",
        width=1280,
        height=720,
    )


if __name__ == "__main__":
    main()
