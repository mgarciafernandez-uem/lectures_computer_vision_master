import torchvision.datasets
import torchvision.transforms
import torch
import torch.nn

from matplotlib import pyplot

class CNN(torch.nn.Module):

    def __init__(self):
        super().__init__()
        
        self.conv = torch.nn.Conv2d(in_channels=1, out_channels=10, kernel_size=28, bias=False)


    def forward(self, x):

        x = self.conv(x)
        x = torch.flatten(x, start_dim=1)
        x = torch.softmax(x, dim=1)

        return x


if __name__ == '__main__':
    mnist = torchvision.datasets.MNIST(root='./mnist/', download=True, train=True, transform=torchvision.transforms.ToTensor())

    mnist_image = list()
    mnist_target = list()

    for image, target in mnist:

        mnist_image.append(image[None, :, :, :])
        mnist_target.append(torch.tensor([[1. if x == target else 0. for x in range(10)]]))

    mnist_image = torch.cat(mnist_image)
    mnist_target = torch.cat(mnist_target)

    split_n = int(len(mnist_image)*0.8)

    train_image = mnist_image[:split_n]
    test_image = mnist_image[split_n:]
    train_target = mnist_target[:split_n]
    test_target = mnist_target[split_n:]
    

    cnn = CNN()
    cnn.eval()

    y = cnn(train_image)
    print(y[:3])
    print(train_target[:3])

    cross_entropy = torch.nn.CrossEntropyLoss()
    optim = torch.optim.Adam(cnn.parameters(), lr=1e-2)


    cnn.train()
    cnn.to('mps')
    train_image = train_image.to('mps')
    train_target = train_target.to('mps')
    test_image = train_image.to('mps')
    test_target = train_target.to('mps')
    
    loss_list = list()
    acc_list = list()
    acc_list_test = list()

    for epoch in range(50):

        running_loss = 0.

        optim.zero_grad()

        cnn.train()
        probs = cnn(train_image)
        loss = cross_entropy(probs, train_target)
        loss.backward()
        optim.step()

        running_loss += loss.item()
        loss_list.append(running_loss)

        _, y_pred = torch.max(probs.data, 1)
        _, target = torch.max(train_target.data , 1)

        acc = (y_pred == target).sum().item() / len(y_pred)

        acc_list.append(acc)

        cnn.eval()
        probs = cnn(test_image)
        _, y_pred = torch.max(probs.data, 1)
        _, target = torch.max(test_target.data , 1)

        acc_test = (y_pred == target).sum().item() / len(y_pred)
        acc_list_test.append(acc_test)


        print(f'Epoch {epoch} - loss {running_loss} - accuracy {acc} - accuracy test {acc_test}')

    state_dict = cnn.state_dict()

    print(state_dict['conv.weight'].shape)

    pyplot.clf()
    pyplot.plot(loss_list)
    pyplot.show()

    pyplot.clf()
    pyplot.plot(acc_list, label='TRAIN')
    pyplot.plot(acc_list_test, label='TEST')
    pyplot.legend()
    pyplot.show()


    pyplot.clf()
    pyplot.subplots(1, 10)
    for i in range(10):
        pyplot.subplot(1, 10, i + 1)
        pyplot.imshow(state_dict['conv.weight'][i, 0].cpu().detach().numpy())
    pyplot.show()