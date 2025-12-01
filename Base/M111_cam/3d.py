import open3d as o3d
pcd = o3d.io.read_point_cloud('my_3d_image_buffer.ply')
o3d.visualization.draw_geometries([pcd])