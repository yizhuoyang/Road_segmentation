import time

import cv2
import numpy as np
import torch
import argparse
import yaml
import math
from torch import Tensor
from torch.nn import functional as F
from pathlib import Path
from torchvision import io
from torchvision import transforms as T
from semseg.models import *
from semseg.datasets import *
from semseg.utils.utils import timer
from semseg.utils.visualize import draw_text
from semseg.datasets.customise import Mydata
from semseg.datasets.ntu import NTU
from rich.console import Console
from PIL import Image
console = Console()


class SemSeg:
    def __init__(self, cfg) -> None:
        # inference device cuda or cpu
        self.device = torch.device(cfg['DEVICE'])

        # get dataset classes' colors and labels
        self.palette = eval(cfg['DATASET']['NAME']).PALETTE
        self.labels = eval(cfg['DATASET']['NAME']).CLASSES

        # initialize the model and load weights and send to device
        self.model = eval(cfg['MODEL']['NAME'])(cfg['MODEL']['BACKBONE'], len(self.palette))
        self.model.load_state_dict(torch.load(cfg['TEST']['MODEL_PATH'], map_location='cpu'))
        # self.model.load_state_dict(torch.load(cfg['TEST']['MODEL_PATH'], map_location='cpu'))
        self.model = self.model.to(self.device)
        self.model.eval()

        # preprocess parameters and transformation pipeline
        self.size = cfg['TEST']['IMAGE_SIZE']
        self.tf_pipeline = T.Compose([
            T.Lambda(lambda x: x / 255),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            T.Lambda(lambda x: x.unsqueeze(0))
        ])

    def preprocess(self, image: Tensor) -> Tensor:
        H, W = image.shape[1:]

        # console.print(f"Original Image Size > [red]{H}x{W}[/red]")
        # scale the short side of image to target size
        scale_factor = self.size[0] / min(H, W)
        nH, nW = round(H*scale_factor), round(W*scale_factor)
        # make it divisible by model stride
        nH, nW = int(math.ceil(nH / 32)) * 32, int(math.ceil(nW / 32)) * 32
        # console.print(f"Inference Image Size > [red]{nH}x{nW}[/red]")
        # resize the image
        image = T.Resize((nH, nW))(image)
        # divide by 255, norm and add batch dim
        image = self.tf_pipeline(image).to(self.device)
        return image

    def postprocess(self, orig_img: Tensor, seg_map: Tensor, overlay: bool) -> Tensor:
        # resize to original image size
        seg_map = F.interpolate(seg_map, size=orig_img.shape[-2:], mode='bilinear', align_corners=True)
        # get segmentation map (value being 0 to num_classes)
        seg_map = seg_map.softmax(dim=1).argmax(dim=1).cpu().to(int)

        # convert segmentation map to color map
        seg_image = self.palette[seg_map].squeeze()
        if overlay:
            seg_image = (orig_img.permute(1, 2, 0) * 0.4) + (seg_image * 0.6)

        # image = draw_text(seg_image, seg_map, self.labels)
        return seg_image

    @torch.inference_mode()
    @timer
    def model_forward(self, img: Tensor) -> Tensor:
        return self.model(img)

    def predict(self, image, overlay: bool) -> Tensor:
        # image = io.read_image(img_fname)
        img = self.preprocess(image)
        seg_map = self.model_forward(img)
        seg_map = self.postprocess(image, seg_map, overlay)
        return seg_map.numpy()


class SegRos(object):
    def __init__(self, predictor):
        self.predictor = predictor
        self.image_subscriber = rospy.Subscriber('/zed2i/zed_node/right_raw/image_raw_color/compressed', CompressedImage, callback=self.image_callback, queue_size=1)
        # self.image_subscriber = rospy.Subscriber('/camera_D455/color/image_raw', Image_type, callback=self.image_callback, queue_size=1)
        self.image_publisher = rospy.Publisher('/image_publish', Image_type, queue_size=1)

    def image_callback(self, msg):
        # image = image_to_numpy(msg)
        start = time.time()
        compressed_data = np.frombuffer(msg.data, np.uint8)
        # Decode the compressed image using OpenCV
        image = cv2.imdecode(compressed_data, cv2.IMREAD_COLOR)

        image = np.transpose(image,(2,0,1))
        # image = np.expand_dims(image,axis=0)
        image = torch.from_numpy(image)
        result_image = self.predictor.predict(
        image,
        cfg['TEST']['OVERLAY'])
        result_image = result_image.astype(np.uint8)
        end = time.time()
        print("infer time:",(end-start)*1000)
        # result_image = self.predictor.visual(outputs[0], img_info, self.predictor.confthre)
        self.image_publisher.publish(numpy_to_image(result_image, encoding='bgr8'))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='configs/ade20k.yaml')
    args = parser.parse_args()

    with open(args.cfg) as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)
    print('Loading model')
    semseg = SemSeg(cfg)
    # segmap = semseg.predict(str(test_file), cfg['TEST']['OVERLAY'])
    import rospy
    print('import success')
    from sensor_msgs.msg import Image as Image_type
    from sensor_msgs.msg import CompressedImage
    from ros_numpy.image import image_to_numpy, numpy_to_image
    rospy.init_node("seg_node")
    yolox_ros = SegRos(predictor=semseg)
    rospy.spin()
