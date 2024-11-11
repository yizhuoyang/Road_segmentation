import numpy as np
from matplotlib import pyplot as plt
import numpy as np
import pydensecrf.densecrf as dcrf
import numpy as np
import cv2
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_labels, create_pairwise_bilateral, create_pairwise_gaussian
from cv2 import imread, imwrite
from scipy.stats import gaussian_kde
import matplotlib.colors as mcolors
import numpy as np
import cv2
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax, create_pairwise_bilateral, create_pairwise_gaussian
from scipy.signal import find_peaks, argrelextrema

def CRFlayer(imginput, maskinput, depthinput,image_xy=10,image_rgb=10,depth_xy=80,depth_rgb=125):
    # 4,2
    # Ensure the mask is binary
    assert set(np.unique(maskinput)).issubset({0, 1}), "The mask should be binary (0 and 1 values only)."

    labels = maskinput.astype(np.int32)
    n_labels = 2  # For binary mask, we have two labels: 0 and 1

    d = dcrf.DenseCRF2D(imginput.shape[1], imginput.shape[0], n_labels)

    # Create the unary potential
    U = unary_from_labels(labels.flatten(), n_labels, gt_prob=0.7, zero_unsure=False)
    d.setUnaryEnergy(U)

    # Create pairwise potentials
    d.addPairwiseGaussian(sxy=(2, 2), compat=10)
    d.addPairwiseBilateral(sxy=(image_xy,image_xy), srgb=(image_rgb, image_rgb, image_rgb), rgbim=imginput, compat=10)

    depthinput = depthinput.astype(np.uint8)

    # Expand depth input to 3D by adding a channel dimension
    depthinput = np.expand_dims(depthinput, axis=2)
    depthinput = np.repeat(depthinput, 3, axis=2)

    # Add pairwise potential using the depth information
    d.addPairwiseBilateral(sxy=(depth_xy,depth_xy), srgb=(depth_rgb, depth_rgb, depth_rgb), rgbim=depthinput, compat=10)

    Q = d.inference(10)
    MAP = np.argmax(Q, axis=0)
    maskoutput = MAP.reshape((imginput.shape[0], imginput.shape[1]))

    return maskoutput



def CRF_depth(mask_depth,S1,S2):

    depthinput = mask_depth.astype(np.uint8)

    # Expand depth input to 3D by adding a channel dimension
    depthinput = np.expand_dims(depthinput, axis=2)
    depthinput = np.repeat(depthinput, 3, axis=2)

    height, width = S1.shape
    labels = np.zeros((height, width), dtype=np.int32)

    # 确定标签
    labels[(S1 == 1) & (S2 == 1)] = 1  # 前景
    labels[(S1 == 0) & (S2 == 0)] = 0  # 背景

    # 不确定标签
    labels[(S1 != S2)] = -1  # 不确定
    # plt.imshow(labels)
    # 创建CRF模型
    d = dcrf.DenseCRF2D(width, height, 2)

    # 设置一元势
    U = unary_from_labels(labels, 2, gt_prob=0.4, zero_unsure=True)
    d.setUnaryEnergy(U)

    # 创建双边势
    pairwise_energy = create_pairwise_bilateral(sdims=(0.2, 0.2), schan=(0.1, 0.1, 0.1),img=depthinput, chdim=2)
    d.addPairwiseEnergy(pairwise_energy, compat=10)

    # 进行推理
    Q = d.inference(5)

    # 得到最优标签
    map = np.argmax(Q, axis=0).reshape((height, width))
    return map


def combine_prob_maps(prob_map1, prob_map2, weight1=0.5*0.8, weight2=0.6):
    """
    结合两个概率图为综合概率图。
    :param prob_map1: 第一张概率图 (numpy array)
    :param prob_map2: 第二张概率图 (numpy array)
    :param weight1: 第一张概率图的权重 (float)
    :param weight2: 第二张概率图的权重 (float)
    :return: 结合后的概率图 (numpy array)
    """
    combined_prob_map = weight1 * prob_map1 + weight2 * prob_map2
    combined_prob_map = np.clip(combined_prob_map, 0, 1)  # 确保概率值在[0, 1]范围内
    return combined_prob_map

def apply_crf(image, combined_prob_map):
    """
    使用CRF优化分割结果
    :param image: 输入图像 (numpy array)
    :param combined_prob_map: 结合后的概率图 (numpy array)
    :return: 分割结果 (numpy array)
    """
    height, width, _ = image.shape

    # 初始化CRF模型
    d = dcrf.DenseCRF2D(width, height, 2)

    # 创建节点能量
    U = unary_from_softmax(combined_prob_map)
    d.setUnaryEnergy(U)

    # 创建对偶能量（高斯对偶项）
    feats = create_pairwise_gaussian(sdims=(5, 5), shape=image.shape[:2])
    d.addPairwiseEnergy(feats, compat=3)

    # 创建对偶能量（双边对偶项）
    feats = create_pairwise_bilateral(sdims=(5, 5), schan=(2, 2, 2), img=image, chdim=2)
    d.addPairwiseEnergy(feats, compat=10)

    # 进行推断
    Q = d.inference(5)
    result = np.argmax(Q, axis=0).reshape((height, width))

    return result

def create_unary_from_prob_map(prob_map):
    """
    将概率图转换为CRF的Unary Energy格式。
    :param prob_map: 结合后的概率图 (numpy array)
    :return: Unary Energy (numpy array)
    """
    # 创建每个像素属于每个类别的概率图
    prob_map_2d = np.stack([1 - prob_map, prob_map], axis=0)
    return prob_map_2d



def vis_mask(image_raw,mask):
    colored_mask = np.zeros_like(image_raw)
    colored_mask[..., 0] = mask * 255  # Red channel
    colored_mask[..., 1] = mask * 0    # Green channel
    colored_mask[..., 2] = mask * 0    # Blue channel

    # Overlay the mask on the image using alpha blending
    alpha = 0.5
    overlay_image = image_raw.copy()
    overlay_image = (overlay_image * (1 - alpha) + colored_mask * alpha).astype(np.uint8)

    # Display the original image and the overlay image
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].imshow(image_raw)
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    axes[1].imshow(overlay_image)
    axes[1].set_title('Image with Overlay')
    axes[1].axis('off')


def save_mask(image_raw, mask):
    colored_mask = np.zeros_like(image_raw)
    colored_mask[..., 0] = mask * 255  # Red channel
    colored_mask[..., 1] = mask * 0    # Green channel
    colored_mask[..., 2] = mask * 0    # Blue channel

    # Overlay the mask on the image using alpha blending
    alpha = 0.5
    overlay_image = image_raw.copy()
    overlay_image = (overlay_image * (1 - alpha) + colored_mask * alpha).astype(np.uint8)

    # Save the overlay image to a file
    plt.imsave('overlay_image.png', overlay_image)

    return overlay_image



def normalize_and_plot_heatmap(mask_depth,image_raw):
    # 正规化处理
    mask_depth_normalized = (mask_depth - mask_depth.min()) / (mask_depth.max() - mask_depth.min())
    mask_depth_normalized = 1 - mask_depth_normalized
    mask_depth_output = mask_depth_normalized.copy()
    # 将原数组中为0的点设置为NaN
    mask_depth_normalized[mask_depth_normalized == 0] = np.nan

    # 创建自定义的颜色映射，设置NaN值为黑色
    cmap = plt.get_cmap('coolwarm')
    cmap.set_bad(color='black')

    plt.imshow(image_raw)
    plt.imshow(mask_depth_normalized, cmap=cmap, interpolation='nearest', alpha=0.5)
    # plt.colorbar()
    # plt.title('Depth to Plane Distance Heatmap')
    plt.show()

    return mask_depth_output

def plot_distance_distribution(distances):
    fig, ax1 = plt.subplots()

    # 直方图
    ax1.hist(distances, bins=100, color='blue', alpha=0.7)
    ax1.set_xlabel('Distance to Plane')
    ax1.set_ylabel('Frequency', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')

    # 曲线图
    ax2 = ax1.twinx()
    density = gaussian_kde(distances)
    xs = np.linspace(min(distances), max(distances), 200)
    ax2.plot(xs, density(xs), color='red')
    ax2.set_ylabel('Density', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    plt.title('Distance Distribution')
    plt.show()


def plot_heatmap(mask_input,image_raw):
    # 正规化处理
    # mask_depth_normalized = (mask_depth - mask_depth.min()) / (mask_depth.max() - mask_depth.min())
    # mask_depth_normalized = 1 - mask_depth_normalized
    # mask_depth_output = mask_depth_normalized.copy()
    # 将原数组中为0的点设置为NaN
    mask = mask_input.copy()
    mask[mask == 0] = np.nan

    # 创建自定义的颜色映射，设置NaN值为黑色
    cmap = plt.get_cmap('coolwarm')
    cmap.set_bad(color='black')

    norm = mcolors.Normalize(vmin=0.1, vmax=1)

    plt.imshow(image_raw)
    plt.imshow(mask, cmap=cmap, norm=norm, interpolation='nearest', alpha=0.5)
    # plt.colorbar()
    # plt.title('Depth to Plane Distance Heatmap')
    plt.show()


def plot_distance_distribution(distances):

    distances = [d for d in distances if d != 0]

    fig, ax1 = plt.subplots()

    # 直方图 (概率密度)
    n, bins, patches = ax1.hist(distances, bins=100, color='blue', alpha=0.7, density=True)
    bin_width = bins[1] - bins[0]
    ax1.set_xlabel('Confidence')
    ax1.set_ylabel('Probability', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')

    # 曲线图
    ax2 = ax1.twinx()
    density = gaussian_kde(distances)
    xs = np.linspace(min(distances), max(distances), 200)
    density_values = density(xs)
    ax2.plot(xs, density_values, color='red')
    ax2.set_ylabel('Density', color='red')
    ax2.tick_params(axis='y', labelcolor='red')

    plt.title('Distance Distribution')
    plt.show()

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
