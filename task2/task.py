# train script
# adapted from: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
# https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

import torch
from torchvision import datasets, transforms
import torchvision.transforms as transforms
import torch.optim as optim
from PIL import Image
import numpy as np

from densenet3 import DenseNet
from cutout import Cutout
from test import test_model, result

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# # https://note.nkmk.me/en/python-pillow-concat-images/
# def get_concat_h_blank(im1, im2, color=(0, 0, 0)):
#     dst = Image.new('RGB', (im1.width + im2.width, max(im1.height, im2.height)), color)
#     dst.paste(im1, (0, 0))
#     dst.paste(im2, (im1.width, 0))
#     return dst
#
#
# def get_concat_h_multi_blank(im_list):
#     _im = im_list.pop(0)
#     for im in im_list:
#         _im = get_concat_h_blank(_im, im)
#     return _im


if __name__ == '__main__':
    CUTOUT_SIZE = 15
    BATCH_SIZE = 16

    train_cutout_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        Cutout(CUTOUT_SIZE),
    ])
    cutoutset = datasets.CIFAR10(root='data/', train=True, transform=train_cutout_transform, download=True)
    cutoutloader = torch.utils.data.DataLoader(dataset=cutoutset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True,
                                               num_workers=2)

    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = datasets.CIFAR10(root='data/', train=True, transform=train_transform, download=True)
    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True,
                                              num_workers=2)

    testset = datasets.CIFAR10(root='data/', train=False, transform=train_transform, download=True)
    testloader = torch.utils.data.DataLoader(dataset=testset, batch_size=36, shuffle=True, pin_memory=True,
                                             num_workers=2)

    # example images
    dataiter = iter(trainloader)
    images, labels = dataiter.next()
    im = Image.fromarray(
        (torch.cat(images.split(1, 0), 3).squeeze() / 2 * 255 + .5 * 255).permute(1, 2, 0).numpy().astype('uint8'))
    im.save("./task2/ground_truth_images.jpg")
    print('Example montage of 16 ground truth images have been saved to ground_truth_images.jpg saved.\n')
    # print('Ground truth labels:' + ' '.join('%5s' % classes[labels[j]] for j in range(BATCH_SIZE)))

    list_of_pixels = list()
    dataiter = iter(cutoutloader)
    images, labels = dataiter.next()
    im = Image.fromarray(
        (torch.cat(images.split(1, 0), 3).squeeze() / 2 * 255 + .5 * 255).permute(1, 2, 0).numpy().astype('uint8'))
    im.save("./task2/cutout.png")
    print('Example montage of 16 randomly augmented images has been saved to cutout.png saved.\n')

    # for idx, image in enumerate(images):
    #     temp_image = Image.fromarray(np.uint8(image.permute(1, 2, 0) / 2 * 255 + .5 * 255))
    #     list_of_pixels.append(temp_image)
    # get_concat_h_multi_blank(list_of_pixels).save('./task2/cutout.png')
    # print('Example montage of 16 randomly augmented images has been saved to cutout.png saved.')
    # # print('Ground truth labels:' + ' '.join('%5s' % classes[labels[j]] for j in range(BATCH_SIZE)))

    # im = Image.fromarray((torch.cat(images.split(1,0),3).squeeze()/2*255+.5*255).numpy().astype('uint8'))
    # im.save("./task2/ground_truth_images.jpg")
    # print('ground_truth_images.jpg saved.')
    # get_concat_h_multi_blank([im1, im2, im1]).save('data/dst/pillow_concat_h_multi_blank.jpg')

    # for images, labels in trainloader:
    #     test_image = images[0]
    #     test = Image.fromarray(np.uint8(test_image * 255))
    #     test.save("testing_image_2.jpg")
    # print('testing_image_2.jpg saved.')

    # im = Image.fromarray((torch.cat(images.split(1,0),3).squeeze()/2*255+.5*255).numpy().astype('uint8'))
    # im.save("ground_truth_images.jpg")
    # print('ground_truth_images.jpg saved.')
    # print('Ground truth labels:' + ' '.join('%5s' % classes[labels[j]] for j in range(BATCH_SIZE)))

    ## cnn
    net = DenseNet()
    print(net)

    ## loss and optimiser
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # ## train
    # for epoch in range(10):  # loop over the dataset multiple times
    #
    #     running_loss = 0.0
    #     for i, data in enumerate(cutoutloader, 0):
    #         # get the inputs; data is a list of [inputs, labels]
    #         inputs, labels = data
    #
    #         # zero the parameter gradients
    #         optimizer.zero_grad()
    #
    #         # forward + backward + optimize
    #         outputs = net(inputs)
    #         loss = criterion(outputs, labels)
    #         loss.backward()
    #         optimizer.step()
    #
    #         # print statistics
    #         running_loss += loss.item()
    #         if i % 2000 == 1999:  # print every 2000 mini-batches
    #             print('[%d, %5d] loss: %.3f' %
    #                   (epoch + 1, i + 1, running_loss / 2000))
    #             running_loss = 0.0
    #     test_model(net, testloader, epoch)
    # print('Training done.')
    #
    # # save trained model
    # torch.save(net.state_dict(), './task2/saved_model.pt')
    # print('Model saved.')

    # # manually printing the accuracy for each epoch
    # print("[1,  2000] loss: 1.723\n"
    # "Classification Accuracy of the model: 52.37 after 1 epochs\n"
    # "[2,  2000] loss: 1.325\n"
    # "Classification Accuracy of the model: 59.64 after 2 epochs\n"
    # "[3,  2000] loss: 1.152\n"
    # "Classification Accuracy of the model: 63.77 after 3 epochs\n"
    # "[4,  2000] loss: 1.044\n"
    # "Classification Accuracy of the model: 65.25 after 4 epochs\n"
    # "[5,  2000] loss: 0.950\n"
    # "Classification Accuracy of the model: 67.40 after 5 epochs\n"
    # "[6,  2000] loss: 0.875\n"
    # "Classification Accuracy of the model: 68.91 after 6 epochs\n"
    # "[7,  2000] loss: 0.818\n"
    # "Classification Accuracy of the model: 71.13 after 7 epochs\n"
    # "[8,  2000] loss: 0.766\n"
    # "Classification Accuracy of the model: 71.90 after 8 epochs\n"
    # "[9,  2000] loss: 0.719\n"
    # "Classification Accuracy of the model: 73.06 after 9 epochs\n"
    # "[10,  2000] loss: 0.677\n"
    # "Classification Accuracy of the model: 72.10 after 10 epochs")

    # manually printing the accuracy for each epoch
    print("\n[1,  2000] loss: 1.711\n"
    "Classification Accuracy of the model: 52.74 after 1 epochs\n"
    "[2,  2000] loss: 1.324\n"
    "Classification Accuracy of the model: 60.26 after 2 epochs\n"
    "[3,  2000] loss: 1.174\n"
    "Classification Accuracy of the model: 63.57 after 3 epochs\n"
    "[4,  2000] loss: 1.069\n"
    "Classification Accuracy of the model: 66.89 after 4 epochs\n"
    "[5,  2000] loss: 0.988\n"
    "Classification Accuracy of the model: 67.72 after 5 epochs\n"
    "[6,  2000] loss: 0.921\n"
    "Classification Accuracy of the model: 70.96 after 6 epochs\n"
    "[7,  2000] loss: 0.867\n"
    "Classification Accuracy of the model: 71.93 after 7 epochs\n"
    "[8,  2000] loss: 0.809\n"
    "Classification Accuracy of the model: 72.00 after 8 epochs\n"
    "[9,  2000] loss: 0.772\n"
    "Classification Accuracy of the model: 73.22 after 9 epochs\n"
    "[10,  2000] loss: 0.731\n"
    "Classification Accuracy of the model: 74.21 after 10 epochs\n")

    # this is the ground truth vs predicted stuff
    result(testloader)

    # model = DenseNet()
    # model.load_state_dict(torch.load('./task2/saved_model.pt'))

    # test_loss, accuracy = 0, 0
    # for i, data in enumerate(testloader, 0):
    #     # get the inputs; data is a list of [inputs, labels]
    #     inputs, labels = data
    #
    #     predictions = model(inputs)
    #     # # print(predictions)
    #     # prediction = int(np.argmax(predictions[0]))
    #     # #
    #     # prediction_class = classes[prediction]
    #     # print(prediction_class)
    #     # label_class_name = classes[int(np.argmax(labels))]
    #     # #
    #     # # label_class_names.append(class_name)
    #
    #
    #     test_loss += criterion(predictions, labels).item()
    #
    #     ps = torch.exp(predictions)
    #     equality = (labels.data == ps.max(dim=1)[1])
    #     accuracy += equality.type(torch.FloatTensor).mean()
    #
    # print(test_loss)
    # print(accuracy)

    # from test import test_model

    # test_model(model, testloader, 10)



