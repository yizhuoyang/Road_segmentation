import torch 
import numpy as np
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import io
from pathlib import Path
from typing import Tuple


class NTU(Dataset):
    """
    num_classes: 19
    # """
    # CLASSES = ['_background_','road','edge']
    #
    #
    # PALETTE = torch.tensor([[0,0,0],[128, 64, 128],[6, 230, 230]])
    #
    # ID2TRAINID = {0: 0, 1: 1, 2:0}


    CLASSES = ['bg','road']

    PALETTE = torch.tensor([[0,0,0],[128, 64, 128]])

    ID2TRAINID = {0: 0, 1: 1}

    def __init__(self, root: str, split: str = 'train', transform = None) -> None:
        super().__init__()
        assert split in ['train', 'val']
        self.transform = transform
        self.n_classes = len(self.CLASSES)
        self.ignore_label = 255

        self.label_map = np.arange(256)
        for id, trainid in self.ID2TRAINID.items():
            self.label_map[id] = trainid

        img_path = Path(root) / 'images' / split
        self.files = list(img_path.rglob('*.png'))
    
        if not self.files:
            raise Exception(f"No images found in {img_path}")
        print(f"Found {len(self.files)} {split} images.")

    def __len__(self) -> int:
        return len(self.files)
    
    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        img_path = str(self.files[index])
        lbl_path = str(self.files[index]).replace('images', 'labels').replace('.png', '.png')
        # print(img_path)
        image = io.read_image(img_path)
        label = io.read_image(lbl_path)
        image = image[:3,:,:]
        label = label[:1,:,:]
        if self.transform:
            image, label = self.transform(image, label)

        return image, self.encode(label.squeeze().numpy()).long()

    def encode(self, label: Tensor) -> Tensor:
        label = self.label_map[label]
        return torch.from_numpy(label)


if __name__ == '__main__':
    from semseg.utils.visualize import visualize_dataset_sample
    visualize_dataset_sample(NTU, '/home/kemove/delta_project/Sementic_segmentation/semantic-segmentation/data/ntu')
