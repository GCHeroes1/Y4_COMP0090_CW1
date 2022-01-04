# https://newbedev.com/how-to-iterate-over-two-dataloaders-simultaneously-using-pytorch


import torch
import torch.optim as optim

from test import test_model

import time
import numpy as np

from itertools import cycle

def training(model, training_dataloaders, testing_loader, epochs, save_path):
    ## loss and optimiser
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    classification_accs = list()
    ## train
    for epoch in range(epochs):  # loop over the dataset multiple times
        start = time.time()
        running_loss = 0.0
        for i, (data1, data2) in enumerate(zip(training_dataloaders[0], training_dataloaders[1]), 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs1, labels1 = data1
            inputs2, labels2 = data2

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs1 = model(inputs1)
            loss = criterion(outputs1, labels1)
            loss.backward()
            optimizer.step()

            # forward + backward + optimize
            outputs2 = model(inputs2)
            loss = criterion(outputs2, labels2)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 400 == 399:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 400))
                running_loss = 0.0
        end = time.time()
        total_time = np.round(end-start, 2)
        classification_acc = test_model(model, testing_loader, epoch)
        classification_accs.append(classification_acc)
        print(f"Classification Accuracy of the model: {classification_acc:.2f} after {str(epoch + 1)} epochs, competed in {str(total_time)} seconds")
    print('Training done.')

    # save trained model
    torch.save(model.state_dict(), save_path)
    print(f'Model saved to {save_path}.\n')
    return save_path, max(classification_accs)
