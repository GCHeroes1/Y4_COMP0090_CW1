# https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
# https://github.com/uoguelph-mlrg/Cutout/blob/master/util/cutout.py

import numpy as np

import torch

from torchvision import datasets, transforms

from PIL import Image
import random

BATCH_SIZE = 128
CUTOUT_SIZE = 20
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

        return img
        # return img.permute(1, 2, 0)


train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    Cutout(CUTOUT_SIZE),
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

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
    # fig = plt.figure(figsize=(8, 5), dpi=600)
    # ax = fig.subplots(4, 8)

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

    # images, labels = next(iter(train_loader))
    # for idx, image in enumerate(images):
    #     ax[idx // 8][idx % 8].imshow(np.uint8(image * 255))
    #     ax[idx // 8][idx % 8].axis('off')
    #
    # plt.show(bbox_inches='tight')

    for images, labels in train_loader:
        test_image = images[0]
        # test = Image.fromarray(np.uint8(test_image * 255))
        test = Image.fromarray(np.uint8(test_image.permute(1, 2, 0) / 2 * 255 + .5 * 255))
        test.save("./task2/train_tf_images_6.jpg")
        break
