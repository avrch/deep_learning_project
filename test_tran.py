import random
import torch

from torchvision.transforms import functional as F


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image):
        for t in self.transforms:
            image = t(image)
        return image

class ToTensor(object):
    def __call__(self, image):
        image = F.to_tensor(image)
        return image
