import matplotlib.pyplot as plt
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, argrelextrema
from scipy.stats import gaussian_kde
from torchvision.transforms import Compose
import cv2
import torch
import open3d as o3d
from skimage.segmentation import felzenszwalb, mark_boundaries
import numpy as np



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

def road_init_with_fit(depth, vis=1, fit=1, normal_threshold=0.2,upper_fit_thred=0.02,lower_fit_thred=0.1,focal_length=500,percentage=0.05,distance_threshold=0.06,distance_thred_depth=0.6):

    height, width = depth.shape
    u, v = np.meshgrid(np.arange(width), np.arange(height))
    cx,cy = width/2,height/2

    Z = 1 / depth * depth.max()
    X = (u - cx) * Z / focal_length
    Y = (v - cy) * Z / focal_length
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

    plane_model, inliers = filtered_pcd.segment_plane(distance_threshold=distance_threshold,
                                                      ransac_n=3,
                                                      num_iterations=1000)

    ground_points = filtered_pcd.select_by_index(inliers)
    ground_points.paint_uniform_color([1.0, 0.0, 0.0])
    ground_indices = filtered_indices[inliers]

    rows, cols = np.unravel_index(ground_indices, depth.shape)
    points_to_mark = list(zip(rows, cols))
    mask_depth = np.zeros(depth.shape, dtype=np.uint8)

    for row, col in points_to_mark:
        mask_depth[row, col] = 1

    if not fit:
        if vis:
            o3d.visualization.draw_geometries([pcd_vis, ground_points])
        return mask_depth, points_to_mark

    else:
        Z = 1/depth*depth.max()*mask_depth
        X = (u - cx) * Z / focal_length
        Y = (v - cy) * Z / focal_length
        mask = np.logical_and.reduce((X >= -5, X <= 5, Z > 1, Z <= 30,Y <= 3))
        points2 = np.stack(( X[mask].flatten(), Z[mask].flatten(),-Y[mask].flatten()), axis=-1)
        pcd2 = o3d.geometry.PointCloud()
        pcd2.points = o3d.utility.Vector3dVector(points2)
        pcd2.paint_uniform_color([0.8, 0.8, 0.8])
        points = points1
        specific_points = points2

        normal, d = fit_plane(specific_points)
        distances,distances_original = points_on_plane(points, normal, d,percentage=percentage,distant_thred=distance_thred_depth)
        return distances.reshape(360,640),distances_original,points_to_mark



def fit_plane(points):
    A = np.c_[points[:, :2], np.ones(points.shape[0])]
    C, _, _, _ = np.linalg.lstsq(A, points[:, 2], rcond=None)
    normal = np.array([-C[0], -C[1], 1])
    normal /= np.linalg.norm(normal)
    d = -C[2]
    return normal, d




def points_on_plane(points, normal, d, auto_thred=1,thred=0.2,percentage=0.08,distant_thred=0.6):

    distances = abs(np.dot(points, normal) + d)
    distances = distances-distances.min()
    distances[distances>=distant_thred]=distant_thred
    distances_original = distances.copy()

    if auto_thred:
        distances_calculate = [d for d in distances if d != 0.6]
        hist, bin_edges = np.histogram(distances_calculate, bins=100, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        smoothed_hist = gaussian_filter1d(hist, sigma=2)
        peak_index = np.argmax(smoothed_hist)
        end_of_peak_index = None
        for i in range(peak_index, len(smoothed_hist)):
            if smoothed_hist[i] < smoothed_hist[peak_index] *percentage:
                end_of_peak_index = i
                break
        if end_of_peak_index is None:
            end_of_peak_index = len(smoothed_hist) - 1
        threshold = bin_centers[end_of_peak_index]
    else:
        threshold=thred

    distances[distances>=threshold]=threshold
    return distances,distances_original


def distance_distribution(distances):
    # 过滤掉 distance 中的 0 值
    distances = [d for d in distances if d != 0]
    density = gaussian_kde(distances)
    xs = np.linspace(min(distances), max(distances), 200)
    density_values = density(xs)
    peaks, _ = find_peaks(density_values)
    if peaks.size > 0:
        first_peak = peaks[-1]
        minima = argrelextrema(density_values, np.less)[0]
        minima_before_first_peak = minima[minima < first_peak]
        if minima_before_first_peak.size > 0:
            start_of_first_wave = xs[minima_before_first_peak[-1]]
        else:
            start_of_first_wave = xs[0]  # If no minimum found, start from the beginning
    else:
        start_of_first_wave = None
    return start_of_first_wave


def segment_superpixels(image, rpd):
    segments = felzenszwalb(image, scale=100, sigma=0.5, min_size=500)
    unique_segments = np.unique(segments)
    rpd_quantiles = np.zeros_like(segments, dtype=float)
    for seg in unique_segments:
        mask = segments == seg
        rpd_quantiles[mask] = np.quantile(rpd[mask], 0.9)
    return rpd_quantiles, segments

# def points_on_plane(points, normal, d, auto_thred=1,thred=0.2,percentage=0.05):
#
#     distances = abs(np.dot(points, normal) + d)
#     distances = distances-distances.min()
#     distances_original = distances.copy()
#     if auto_thred:
#         threshold=0
#     else:
#         threshold=thred
#     distances[distances>=threshold]=threshold
#     return distances,distances_original



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


