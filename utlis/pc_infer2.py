import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision.transforms import Compose
from depth_anything.dpt import DepthAnything
from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet
import cv2
import torch
import open3d as o3d
import numpy as np

def depth_infer(image,depth_anything,device):
    h, w = image.shape[0],image.shape[1]
    transform = Compose([
        Resize(
            width=518,
            height=518,
            resize_target=False,
            keep_aspect_ratio=True,
            ensure_multiple_of=14,
            resize_method='lower_bound',
            image_interpolation_method=cv2.INTER_CUBIC,
        ),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet(),
    ])

    image = transform({'image': image})['image']
    image = torch.from_numpy(image).unsqueeze(0).to(device)

    with torch.no_grad():
        depth = depth_anything(image)

    depth = F.interpolate(depth[None], (h, w), mode='bilinear', align_corners=False)[0, 0]
    depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0

    depth = depth.cpu().numpy().astype(np.uint8)
    # depth[depth<0.1]=255
    depth[depth<=10]=10
    return depth

def depth_infer2(image,depth_anything,device):
    h, w = image.shape[0],image.shape[1]
    transform = Compose([
        Resize(
            width=518,
            height=518,
            resize_target=False,
            keep_aspect_ratio=True,
            ensure_multiple_of=14,
            resize_method='lower_bound',
            image_interpolation_method=cv2.INTER_CUBIC,
        ),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet(),
    ])

    image = transform({'image': image})['image']
    image = torch.from_numpy(image).unsqueeze(0).to(device)

    with torch.no_grad():
        depth = depth_anything(image)

    depth = F.interpolate(depth[None], (h, w), mode='bilinear', align_corners=False)[0, 0]
    depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0

    depth = depth.cpu().numpy().astype(np.uint8)
    # depth[depth<0.1]=255
    depth[depth<=10]=10
    return depth

def road_init(x, y, depth, vis=1, normal_threshold=0.2):
    # 计算Z，X和Y坐标
    Z = 1 / depth * 255
    X = x * Z
    Y = y * Z
    points1 = np.stack((X.flatten(), Z.flatten(), -Y.flatten()), axis=-1)

    mask = np.logical_and.reduce((X >= -5, X <= 5, Z > 0.1, Z <= 30))
    points_vis = np.stack((X[mask].flatten(), Z[mask].flatten(), -Y[mask].flatten()), axis=-1)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points1)

    pcd_vis = o3d.geometry.PointCloud()
    pcd_vis.points = o3d.utility.Vector3dVector(points_vis)

    # 仅保留深度值不为10的点进行平面分割
    valid_mask = (depth.flatten() != 10)
    valid_points = points1[valid_mask]
    valid_indices = np.where(valid_mask)[0]  # 保存有效点的索引

    pcd_valid = o3d.geometry.PointCloud()
    pcd_valid.points = o3d.utility.Vector3dVector(valid_points)

    # 法向量估计
    pcd_valid.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    # 获取法向量并过滤掉非水平面的点
    normals = np.asarray(pcd_valid.normals)
    vertical_normals_mask = np.abs(normals[:, 2]) > normal_threshold
    filtered_points = valid_points[vertical_normals_mask]
    filtered_indices = valid_indices[vertical_normals_mask]  # 保存过滤后的点的索引

    filtered_pcd = o3d.geometry.PointCloud()
    filtered_pcd.points = o3d.utility.Vector3dVector(filtered_points)

    # 使用RANSAC分割平面
    plane_model, inliers = filtered_pcd.segment_plane(distance_threshold=0.01,
                                                      ransac_n=3,
                                                      num_iterations=1000)

    ground_points = filtered_pcd.select_by_index(inliers)
    ground_points.paint_uniform_color([1.0, 0.0, 0.0])

    if vis:
        o3d.visualization.draw_geometries([pcd_vis, ground_points])

    # 映射分割后的点回到原始图像坐标系
    ground_indices = filtered_indices[inliers]
    rows, cols = np.unravel_index(ground_indices, x.shape)
    points_to_mark = list(zip(rows, cols))
    mask_depth = np.zeros(x.shape, dtype=np.uint8)
    for row, col in points_to_mark:
        mask_depth[row, col] = 1

    return mask_depth, points_to_mark

def road_init_with_fit(x, y, depth, vis=1, fit=1, normal_threshold=0.2,upper_fit_thred=0.02,lower_fit_thred=0.1):
    Z = 1 / depth * 255
    X = x * Z
    Y = y * Z
    points1 = np.stack((X.flatten(), Z.flatten(), -Y.flatten()), axis=-1)

    mask = np.logical_and.reduce((X >= -5, X <= 5, Z > 0.1, Z <= 30))
    points_vis = np.stack((X[mask].flatten(), Z[mask].flatten(), -Y[mask].flatten()), axis=-1)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points1)

    pcd_vis = o3d.geometry.PointCloud()
    pcd_vis.points = o3d.utility.Vector3dVector(points_vis)

    valid_mask = (depth.flatten() != 10)
    valid_points = points1[valid_mask]
    valid_indices = np.where(valid_mask)[0]

    pcd_valid = o3d.geometry.PointCloud()
    pcd_valid.points = o3d.utility.Vector3dVector(valid_points)

    pcd_valid.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    normals = np.asarray(pcd_valid.normals)
    vertical_normals_mask = np.abs(normals[:, 2]) > normal_threshold
    filtered_points = valid_points[vertical_normals_mask]
    filtered_indices = valid_indices[vertical_normals_mask]

    filtered_pcd = o3d.geometry.PointCloud()
    filtered_pcd.points = o3d.utility.Vector3dVector(filtered_points)

    plane_model, inliers = filtered_pcd.segment_plane(distance_threshold=0.01,
                                                      ransac_n=3,
                                                      num_iterations=1000)

    ground_points = filtered_pcd.select_by_index(inliers)
    ground_points.paint_uniform_color([1.0, 0.0, 0.0])
    ground_indices = filtered_indices[inliers]

    rows, cols = np.unravel_index(ground_indices, x.shape)
    points_to_mark = list(zip(rows, cols))
    mask_depth = np.zeros(x.shape, dtype=np.uint8)
    for row, col in points_to_mark:
        mask_depth[row, col] = 1

    if not fit:
        if vis:
            o3d.visualization.draw_geometries([pcd_vis, ground_points])
        return mask_depth, points_to_mark

    else:
        Z = 1/depth*255*1*mask_depth
        X = x * Z
        Y = y * Z
        mask = np.logical_and.reduce((X >= -5, X <= 5, Z > 1, Z <= 30,y > -1, y <= 3))
        points2 = np.stack(( X[mask].flatten(), Z[mask].flatten(),-Y[mask].flatten()), axis=-1)
        pcd2 = o3d.geometry.PointCloud()
        pcd2.points = o3d.utility.Vector3dVector(points2)
        pcd2.paint_uniform_color([0.8, 0.8, 0.8])
        points = points1
        specific_points = points2

        normal, d = fit_plane(specific_points)
        plane_points, plane_indices = points_on_plane(points, normal, d,upper_threshold=upper_fit_thred,lower_threshold=lower_fit_thred)

        plane_pcd = o3d.geometry.PointCloud()
        plane_pcd.points = o3d.utility.Vector3dVector(plane_points)

        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])

        center = specific_points.mean(axis=0)
        u = specific_points[1] - specific_points[0]
        v = specific_points[2] - specific_points[0]
        u /= np.linalg.norm(u)
        v /= np.linalg.norm(v)
        corner = center - u - v
        corners = [corner, corner + u * 2, corner + u * 2 + v * 2, corner + v * 2]
        faces = [[0, 1, 2], [0, 2, 3]]

        rows, cols = np.unravel_index(plane_indices, X.shape)
        points_to_mark = list(zip(rows, cols))
        mask_depth = np.zeros(X.shape, dtype=np.uint8)
        for row, col in points_to_mark:
            mask_depth[row, col] = 1
        if vis:
            vertices = o3d.utility.Vector3dVector(corners)
            triangles = o3d.utility.Vector3iVector(faces)
            plane_mesh = o3d.geometry.TriangleMesh(vertices, triangles)
            plane_pcd.paint_uniform_color([0, 1, 0])

            o3d.visualization.draw_geometries([pcd_vis, plane_pcd, mesh_frame, plane_mesh])
        return mask_depth, points_to_mark



def fit_plane(points):
    A = np.c_[points[:, :2], np.ones(points.shape[0])]
    C, _, _, _ = np.linalg.lstsq(A, points[:, 2], rcond=None)
    normal = np.array([-C[0], -C[1], 1])
    normal /= np.linalg.norm(normal)
    d = -C[2]
    return normal, d


def points_on_plane(points, normal, d, upper_threshold=0.03, lower_threshold=0.1):
    distances = np.dot(points, normal) + d

    indices = np.where((distances < upper_threshold) & (distances > -lower_threshold))[0]

    return points[indices], indices

# def road_init(x, y, depth, vis=1,thred=0.005):
#     Z = 1 / depth * 255
#     X = x * Z
#     Y = y * Z
#     points1 = np.stack((X.flatten(), Z.flatten(), -Y.flatten()), axis=-1)
#
#     mask = np.logical_and.reduce((X >= -5, X <= 5, Z > 0.1, Z <= 30))
#     points_vis = np.stack((X[mask].flatten(), Z[mask].flatten(), -Y[mask].flatten()), axis=-1)
#
#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(points1)
#
#     pcd_vis = o3d.geometry.PointCloud()
#     pcd_vis.points = o3d.utility.Vector3dVector(points_vis)
#
#     # 仅保留深度值不为10的点进行平面分割
#     valid_mask = (depth.flatten() != 10)
#     valid_points = points1[valid_mask]
#
#     pcd_valid = o3d.geometry.PointCloud()
#     pcd_valid.points = o3d.utility.Vector3dVector(valid_points)
#
#
#     plane_model, inliers = pcd_valid.segment_plane(distance_threshold=thred,
#                                                    ransac_n=3,
#                                                    num_iterations=1000)
#
#     inlier_points = valid_points[inliers]
#
#     ground_points = o3d.geometry.PointCloud()
#     ground_points.points = o3d.utility.Vector3dVector(inlier_points)
#     ground_points.paint_uniform_color([1.0, 0.0, 0.0])
#
#     non_ground_points = pcd_valid.select_by_index(inliers, invert=True)
#
#     if vis:
#         o3d.visualization.draw_geometries([pcd_vis, ground_points])
#
#     inlier_indices = np.where(valid_mask)[0][inliers]
#     rows, cols = np.unravel_index(inlier_indices, X.shape)
#     points_to_mark = list(zip(rows, cols))
#
#     mask_depth = np.zeros(X.shape, dtype=np.uint8)
#     for row, col in points_to_mark:
#         mask_depth[row, col] = 1
#
#     return mask_depth, points_to_mark
#


