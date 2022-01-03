import torch
from torchvision import datasets, transforms
import torchvision.transforms as transforms
import torch.optim as optim
import numpy as np

from densenet3 import DenseNet
from cutout import Cutout
from test import test_model, result, test_model_loss
from dense_training import training
from resnet import ResNet18
from wide_resnet import WideResNet
from resnet_training import resnet_training

EPOCHS = 10

def three_fold_dataset(data):
    indices = np.arange(0, len(data))  # build an array = np.asrray( [x for x in range(len(data))])
    np.random.shuffle(indices)  # shuffle the indicies

    fold_size = int(len(data) / 3)

    first_fold_loader = torch.utils.data.DataLoader(data, batch_size=BATCH_SIZE, shuffle=False,
                                                    sampler=torch.utils.data.SubsetRandomSampler(indices[:fold_size]))

    second_fold_loader = torch.utils.data.DataLoader(data, batch_size=BATCH_SIZE, shuffle=False,
                                                     sampler=torch.utils.data.SubsetRandomSampler(
                                                         indices[fold_size:-fold_size]))

    third_fold_loader = torch.utils.data.DataLoader(data, batch_size=BATCH_SIZE, shuffle=False,
                                                    sampler=torch.utils.data.SubsetRandomSampler(
                                                        indices[-fold_size:]))

    print(f"the dataset size for the first fold is {str(len(first_fold_loader) * BATCH_SIZE)}")
    print(f"the dataset size for the second fold is {str(len(second_fold_loader) * BATCH_SIZE)}")
    print(f"the dataset size for the third fold is {str(len(third_fold_loader) * BATCH_SIZE)} \n")
    # remaining_fold_size = len(data) - fold_size
    # first_fold, second_third_fold = torch.utils.data.random_split(data, [fold_size, remaining_fold_size])
    # print(first_fold.size())
    # remaining_fold_size = remaining_fold_size // 2
    # second_fold, third_fold = torch.utils.data.random_split(second_third_fold,
    #                                                         [remaining_fold_size, remaining_fold_size])
    # print(second_fold.size())
    return first_fold_loader, second_fold_loader, third_fold_loader


def three_fold_validation(raw_data, aug_data):
    fold_raw_dataloaders = three_fold_dataset(raw_data)  # split the dataset into 3 parts
    fold_aug_dataloaders = three_fold_dataset(aug_data)
    raw_predictor = ['', -1]  # update as we find better predictors
    aug_predictor = ['', -1]  # update as we find better predictors

    for i in range(3):  # three fold validation
        temp_raw_model = DenseNet()
        temp_aug_model = DenseNet()
        training_raw_dataloaders = list()
        training_aug_dataloaders = list()
        for j in range(3):
            if j != i:
                training_raw_dataloaders.append(fold_raw_dataloaders[j])
                training_aug_dataloaders.append(fold_aug_dataloaders[j])

        testing_loader = fold_raw_dataloaders[i]

        raw_save_path, raw_error = training(temp_raw_model, training_raw_dataloaders, testing_loader,
                                                            EPOCHS, f'./task3/temp_model_{str(i)}_raw_{str(EPOCHS)}.pt')
        aug_save_path, aug_error = training(temp_aug_model, training_aug_dataloaders, testing_loader,
                                                            EPOCHS, f'./task3/temp_model_{str(i)}_aug_{str(EPOCHS)}.pt')
        if raw_error >= raw_predictor[1]:
            raw_predictor[0], raw_predictor[1] = raw_save_path, raw_error
        if aug_error >= aug_predictor[1]:
            aug_predictor[0], aug_predictor[1] = aug_save_path, aug_error
    return raw_predictor, aug_predictor


if __name__ == '__main__':
    print("I will be doing an ablation study on the difference between training with and without the cutout data "
          "augmentation algorithm described in Task 2")

    CUTOUT_SIZE = 15
    BATCH_SIZE = 16

    train_cutout_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        Cutout(CUTOUT_SIZE),
    ])

    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    raw_dataset = datasets.CIFAR10(root='data/', train=True, transform=train_transform, download=True)

    cutout_dataset = datasets.CIFAR10(root='data/', train=True, transform=train_cutout_transform, download=True)

    # cutoutloader = torch.utils.data.DataLoader(dataset=cutout_dataset, batch_size=BATCH_SIZE, shuffle=True,
    # pin_memory=True, num_workers=2)

    # datasetloader = torch.utils.data.DataLoader(dataset=raw_dataset, batch_size=BATCH_SIZE, shuffle=True,
    # pin_memory=True, num_workers=2)

    # testset = datasets.CIFAR10(root='data/', train=False, transform=train_transform, download=True)
    # testloader = torch.utils.data.DataLoader(dataset=testset, batch_size=36, shuffle=True, pin_memory=True,
    #                                          num_workers=2)

    # print(len(raw_dataset))
    # print(len(cutout_dataset))
    development_size = int(0.8 * len(raw_dataset))
    holdout_test_size = len(raw_dataset) - development_size

    development_raw_dataset, holdout_test_dataset = torch.utils.data.random_split(raw_dataset,
                                                                                  [development_size, holdout_test_size])
    development_cutout_dataset, holdout_cutout_test_dataset = torch.utils.data.random_split(cutout_dataset,
                                                                                            [development_size,
                                                                                             holdout_test_size])

    print(three_fold_validation(development_raw_dataset, development_cutout_dataset))

    # so im going to do three fold validation on the development set, and with that pick the subset of the dataset
    # that gave the lowest error, and then use that
    # for testing on the holdout set that we've been saving

    # # only ever needs to be ran once to train the models initially
    # development_loader = torch.utils.data.DataLoader(dataset=development_raw_dataset, batch_size=BATCH_SIZE,
    #                                                  shuffle=True, pin_memory=True, num_workers=2)
    #
    # model_resnet18 = ResNet18(num_classes=int(10))
    # model_wideresnet = WideResNet(depth=int(28), num_classes=int(10), widen_factor=int(10), dropRate=float(0.3))
    #
    # model_resnet18, resnet_save_path = resnet_training(model_resnet18, development_loader, EPOCHS,
    #                                                    f'./task3/resnet18_model.pt')
    # model_wideresnet, wide_resnet_save_path = resnet_training(model_wideresnet, development_loader, EPOCHS,
    #                                                           f'./task3/wide_resnet_model.pt')



    holdout_test_loader = torch.utils.data.DataLoader(dataset=holdout_test_dataset, batch_size=BATCH_SIZE,
                                                     shuffle=False, pin_memory=True, num_workers=2)

    # model_densenet_raw = DenseNet()
    # model_densenet_raw.load_state_dict(torch.load(f'./task3/temp_model_{str(1)}_raw.pt'))
    # raw_classification_acc, raw_loss = test_model_loss(model_densenet_raw, holdout_test_loader, EPOCHS)
    # print(f"DenseNet3 trained on Cifar10 without data augmentation, with cross validation, achieved a classification "
    #       f"accuracy of {str(raw_classification_acc)} and a loss of {str(np.round(raw_loss, 3))}")
    #
    #
    # model_densenet_aug = DenseNet()
    # model_densenet_aug.load_state_dict(torch.load(f'./task3/temp_model_{str(1)}_aug.pt'))
    # aug_classification_acc, aug_loss = test_model_loss(model_densenet_aug, holdout_test_loader, EPOCHS)
    # print(f"DenseNet3 trained on Cifar10 with data augmentation, with cross validation, achieved a classification "
    #       f"accuracy of {str(aug_classification_acc)} and a loss of {str(np.round(aug_loss, 3))}")
    #
    # model_resnet18 = ResNet18(num_classes=int(10))
    # model_resnet18.load_state_dict(torch.load(f'./task3/resnet18_model.pt'))
    # resnet18_classification_acc, resnet18_loss = test_model_loss(model_densenet_aug, holdout_test_loader, EPOCHS)
    # print(f"ResNet18 trained on Cifar10 achieved a classification accuracy of {str(resnet18_classification_acc)} "
    #       f"and a loss of {str(np.round(resnet18_loss, 3))}")
    #
    # model_wideresnet = WideResNet(depth=int(28), num_classes=int(10), widen_factor=int(10), dropRate=float(0.3))
    # model_wideresnet.load_state_dict(torch.load(f'./task3/wide_resnet_model.pt'))
    # wideresnet_classification_acc, wideresnet_loss = test_model_loss(model_densenet_aug, holdout_test_loader, EPOCHS)
    # print(f"WideResNet trained on Cifar10 achieved a classification accuracy of {str(wideresnet_classification_acc)} "
    #       f"and a loss of {str(np.round(wideresnet_lossm 3))}")