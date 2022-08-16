import albumentations
import torch
import numpy as np
from PIL import Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


class ImageDataset:
    def __init__(self, image_paths, labels, resize= None):

        #resize -> (height, width)
        self.image_paths = image_paths
        self.labels = labels
        self.resize = resize
        self.aug = albumentations.Compose([albumentations.Normalize(always_apply=True)])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        label = self.labels[idx]

        if self.resize is not None:
            image = image.resize((self.resize[1], self.resize[0]), resample=Image.BILINEAR)

        image = np.array(image)
        augmented = self.aug(image = image)
        image = augmented["image"]

        image = np.transpose(image, (2, 0, 1)).astype(np.float32)

        return {
            "images" : torch.tensor(image, dtype = torch.float),
            "labels" : torch.tensor(label, dtype = torch.long),
        }
        


    