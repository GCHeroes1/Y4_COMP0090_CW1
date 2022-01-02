# test script
# adapted from: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
# https://github.com/uoguelph-mlrg/Cutout/blob/master/train.py
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image

from densenet3 import DenseNet


def test_model(model, testloader):
    model.eval()
    num_correct, total = 0, 0
    for i, data in enumerate(testloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        with torch.no_grad():
            outputs = model(inputs)

        pred = torch.max(outputs, 1)[1]

        num_correct += (pred == labels).sum()
        total += labels.size(0)
    print(f"Test Accuracy of the model: {float(num_correct) / float(total) * 100:.2f} after 10 epochs")
    model.train()
    return


if __name__ == '__main__':
    ## cifar-10 dataset
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = 36

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
    dataiter = iter(testloader)
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


    ## load the trained model
    model = DenseNet()
    model.load_state_dict(torch.load('./task2/saved_model.pt'))

    ## inference
    images, labels = dataiter.next()
    outputs = model(images)
    predicted = torch.max(outputs, 1)[1]

    print('Ground-truth: ', ' '.join('%5s' % classes[labels[j]] for j in range(batch_size)))
    print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(batch_size)))

    # save to images
    im = Image.fromarray((torch.cat(images.split(1,0),3).squeeze()/2*255+.5*255).permute(1,2,0).numpy().astype('uint8'))
    im.save("./task2/test_pt_images.jpg")
    print('test_pt_images.jpg saved.')

    test_model(model, testloader)
