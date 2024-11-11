import cv2
import sys

import torch

sys.path.append('/home/kemove/delta_project/Sementic_segmentation/Road_segmentation/segment-anything')
from segment_anything import sam_model_registry, SamPredictor
import numpy as np
import random
from sklearn.cluster import KMeans
from robust_segment_anything import SamPredictor, sam_model_registry
from robust_segment_anything import SamAutomaticMaskGenerator, sam_model_registry,build_sam_vit_l
from robust_segment_anything.utils.transforms import ResizeLongestSide



def sample_spread_out_points(points, num_samples):
    kmeans = KMeans(n_clusters=num_samples, random_state=0, n_init=10).fit(points)
    return kmeans.cluster_centers_

def sam_predict(image,points_to_mark,predictor,opt,use_robust=0,num_groups = 200,num_points_per_group = 2,use_mean=1,p=0.8):

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
        if use_robust:
            mask = robust_sam_predict(predictor,image,input_point,opt)
        else:
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

    return  final_mask[0]


def robust_sam_predict(model,image,prompt,opt):
    sam_transform = ResizeLongestSide(model.image_encoder.img_size)
    image_t = torch.tensor(image, dtype=torch.uint8).unsqueeze(0).to(opt.gpu)
    image_t = torch.permute(image_t, (0, 3, 1, 2))
    image_t_transformed = sam_transform.apply_image_torch(image_t.float())
    data_dict = {}

    input_label = torch.Tensor(np.ones(prompt.shape[0])).to(opt.gpu)
    point_t = torch.Tensor(prompt).to(opt.gpu)
    data_dict['image'] = image_t_transformed
    data_dict['point_coords'] = sam_transform.apply_coords_torch(point_t, image_t.shape[-2:]).unsqueeze(0)
    data_dict['point_labels'] = input_label.unsqueeze(0)
    data_dict['original_size'] = image_t.shape[-2:]
    with torch.no_grad():
        batched_output = model.predict(opt, [data_dict], multimask_output=False, return_logits=False)

    output_mask = batched_output[0]['masks']
    return output_mask[0].cpu().numpy()




