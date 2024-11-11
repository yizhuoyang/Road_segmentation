import cv2
import sys
import torch
from utlis.pc_infer_fusion import distance_distribution
from utlis.visualiaztion import plot_distance_distribution
sys.path.append('../')
from segment_anything import sam_model_registry, SamPredictor
import numpy as np
import random
from sklearn.cluster import KMeans
from utlis.visualiaztion import *


def sample_spread_out_points(points, num_samples):
    kmeans = KMeans(n_clusters=num_samples, random_state=0, n_init=10).fit(points)
    return kmeans.cluster_centers_

def sam_predict(image,points_to_mark,predictor,opt,device,use_robust=0,num_groups = 200,num_points_per_group = 2,use_mean=1,p=0.8,condidence=0.2,iou=0.9):

    if not use_robust:
        predictor.set_image(image)

    points_to_mark = np.array(points_to_mark)

    selected_points_groups = []
    for _ in range(num_groups):
        selected_indices = np.random.choice(len(points_to_mark), num_points_per_group, replace=False)
        selected_points = points_to_mark[selected_indices]
        swapped_points = selected_points[:, [1, 0]]
        selected_points_groups.append(swapped_points)

    masks = []
    areas = []
    for points in selected_points_groups:
        spread_out_points = sample_spread_out_points(points, num_points_per_group)
        input_point = spread_out_points
        input_label = np.ones(num_points_per_group)

        mask, _, _ = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=False,
        )

        masks.append(mask*1.0)
        areas.append(np.sum(mask))

    max_area_idx = np.argmax(areas)
    min_area_idx = np.argmin(areas)
    filtered_masks = [mask for i, mask in enumerate(masks) if i != max_area_idx and i != min_area_idx]

    if use_mean:
        mean_mask = np.mean(filtered_masks, axis=0)
        final_mask = mean_mask > np.mean(mean_mask)
    else:
        final_mask = np.mean(filtered_masks, axis=0) > p

    return  final_mask[0],np.mean(filtered_masks, axis=0)

def fustion_info(mean_mask,mask_depth_normalized,image_raw):

    a = mean_mask.copy()
    valid = np.array([d for d in a.flatten() if d != 0])
    # a[a!=0]=1
    # a[a>=valid.mean()]=1
    a[a>=0.2]=1
    a[a!=1]=0

    b = mask_depth_normalized.copy()
    valid = np.array([d for d in b.flatten() if d != 0])
    b[b>=b.mean()]=1
    b[b!=1]=0

    c = a*b
    c = 1-c
    d = ((mean_mask[0]+0.2*mask_depth_normalized)*c)
    t = distance_distribution(d.flatten())
    d[d>=t]=1
    d[d!=1]=0
    f = (1-c)+d
    return f
#
# def fustion_info2(mean_mask,mask_depth_normalized):
#
#     mask_depth_normalized[mask_depth_normalized>0.6]=0.4
#     mask_depth_normalized[mask_depth_normalized<0.1]=0
#     data = mask_depth_normalized
#     non_zero_data = data[data != 0]
#     target_mean = 0.8
#     target_variance = 0.01
#     adjusted_non_zero_data = (non_zero_data - np.mean(non_zero_data)) / np.std(non_zero_data) * np.sqrt(target_variance) + target_mean
#     adjusted_data = np.copy(data)
#     adjusted_data[adjusted_data != 0] = adjusted_non_zero_data
#     mask_depth_normalized = adjusted_data
#
#     a = mean_mask.copy()
#     a[a>0.2]=1
#
#     b = mask_depth_normalized.copy()*a[0]
#     c = mean_mask+b
#     c[c>=0.5]=1
#     c[c!=1]=0
#     return c



def fustion_info2(mean_mask,mask_depth_normalized,image_raw):
    a = mean_mask.copy()
    x = a[a!=0]
    a[a>0.1]=1

    b = mask_depth_normalized.copy()*a[0]
    unary = create_unary_from_prob_map(0.5*mean_mask+0.5*mask_depth_normalized)
    segmented_image = apply_crf(image_raw, unary)
    return segmented_image
