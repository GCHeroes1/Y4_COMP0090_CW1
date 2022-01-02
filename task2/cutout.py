# • Implement a data augmentation function cutout, using the Cutout algorithm.
# o Use square masks with variable size and location. [2]
# o Add an additional parameter s, such that the mask size can be uniformly sampled from [0, s].
# [3]
# o Location should be sampled uniformly in the image space. N.B. care needs to be taken around
# the boundaries, so the sampled mask maintains its size. [3]
# o Visualise your implementation, by saving to a PNG file “cutout.png”, a montage of 16 images
# with randomly augmented images that are about to be fed into network training. [3]
# o Add Cutout into the network training. [3]

import pdb
import argparse
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import MultiStepLR

from torchvision.utils import make_grid
from torchvision import datasets, transforms

from PIL import Image
import random

BATCH_SIZE = 128
CUTOUT_SIZE = 10
NUM_CLASSES = 10


class Cutout(object):
    def __init__(self, length):
        self.length = random.randint(0, length)

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with hole of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        length_half = self.length // 2
        offset = 1 if self.length % 2 == 0 else 0

        mask = np.ones((h, w), np.float32)

        cxmin, cxmax = length_half, w + offset - length_half
        cymin, cymax = length_half, h + offset - length_half

        x = np.random.randint(cxmin, cxmax)
        y = np.random.randint(cymin, cymax)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img.permute(1, 2, 0)


# def cutout(size, mask_color=(0, 0, 0)):
#     mask_size = random.randint(0, size)
#
#     mask_size_half = mask_size // 2
#     offset = 1 if mask_size % 2 == 0 else 0
#
#     def _cutout(image):
#         image = np.asarray(image).copy()
#
#         h, w = image.shape[:2]
#
#         # if (cx + mask_size_half + offset) > w:
#         #     cxmin, cxmax = w - mask_size, w
#         # else:
#         #     cxmin, cxmax = 0, mask_size
#         # if (cy + mask_size_half + offset) > h:
#         #     cymin, cymax = h - mask_size, h
#         # else:
#         #     cymin, cymax = 0, mask_size
#
#         cxmin, cxmax = mask_size_half, w + offset - mask_size_half
#         cymin, cymax = mask_size_half, h + offset - mask_size_half
#
#         cx = np.random.randint(cxmin, cxmax)
#         cy = np.random.randint(cymin, cymax)
#         xmin, xmax = cx - mask_size_half, xmin + mask_size
#         ymin, ymax = cy - mask_size_half, ymin + mask_size
#         # xmax = xmin + mask_size
#         # ymax = ymin + mask_size
#         xmin, xmax = max(0, xmin), min(w, xmax)
#         ymin, ymax = max(0, ymin), min(h, ymax)
#         # xmax = min(w, xmax)
#         # ymax = min(h, ymax)
#         image[ymin:ymax, xmin:xmax] = mask_color
#         return image
#
#     return _cutout

# im = Image.fromarray(tf.concat([train_images[i,...] for i in range(num_images)],1).numpy())
# im.save("train_tf_images.jpg")

mean = np.array([0.4914, 0.4822, 0.4465])
std = np.array([0.2470, 0.2435, 0.2616])

train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
    Cutout(CUTOUT_SIZE),
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std)])

train_dataset = datasets.CIFAR10(root='data/',
                                 train=True,
                                 transform=train_transform,
                                 download=True)

test_dataset = datasets.CIFAR10(root='data/',
                                train=False,
                                transform=test_transform,
                                download=True)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=BATCH_SIZE,
                                           shuffle=True,
                                           pin_memory=True,
                                           num_workers=2)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=BATCH_SIZE,
                                          shuffle=False,
                                          pin_memory=True,
                                          num_workers=2)

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(8, 5), dpi=600)
    ax = fig.subplots(4, 8)

    # images, labels = next(iter(train_loader))
    # test_image = Cutout(images[0], CUTOUT_SIZE)
    # test = Image.fromarray(np.uint8(test_image.permute(1, 2, 0) * 255))
    # test.save("train_tf_images.jpg")
    # im = Image.fromarray(torch.cat([test_image[i, ...] for i in range(images)], 1).numpy())
    # im.save("train_tf_images.jpg")
    # plt.imshow(np.transpose(test_image, (1, 2, 0)).astype('uint8'))
    # for images, labels in train_loader:
    #     test_image = images[0]
    #     test = Image.fromarray(np.uint8(test_image * 255))
    #     test.save("train_tf_images.jpg")
    #     # plt.imshow(np.uint8(test_image * 255))
    #     #     images = cutout(test_image, CUTOUT_SIZE)
    #     #     labels = labels.cuda()
    #     #     # plt.show(images)
    #     break

    images, labels = next(iter(train_loader))
    for idx, image in enumerate(images):
        ax[idx // 8][idx % 8].imshow(np.uint8(image * 255))
        ax[idx // 8][idx % 8].axis('off')

    plt.show(bbox_inches='tight')

