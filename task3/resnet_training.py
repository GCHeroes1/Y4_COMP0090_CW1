import torch
import torch.optim as optim

from test import test_model

import time
import numpy as np

from itertools import cycle

def resnet_training(model, training_dataloader, epochs, save_path):
    ## loss and optimiser
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # classification_accs = list()
    ## train
    for epoch in range(epochs):  # loop over the dataset multiple times
        start = time.time()
        running_loss = 0.0
        for i, data in enumerate(training_dataloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
        end = time.time()
        total_time = np.round(end-start, 2)
        # classification_acc = test_model(model, testing_loader, epoch)
        # classification_accs.append(classification_acc)
        print(f"epoch {str(epoch + 1)} competed in {str(total_time)} seconds")
    print('Training done.')

    # save trained model
    torch.save(model.state_dict(), save_path)
    print(f'Model saved to {save_path}.')
    return model, save_path
