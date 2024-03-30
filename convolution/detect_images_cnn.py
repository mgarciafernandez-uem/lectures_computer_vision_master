import torchvision.datasets
import torchvision.transforms
import torchvision.io
import torch.utils.data
import torch.utils.data.dataloader
import torch.nn
import torch.optim

from matplotlib import pyplot

class CNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels=3, out_channels=3, kernel_size=10, bias=False)

    def forward(self, x):
        x = self.conv(x)

        return x
    

if __name__ == '__main__':
    image = torchvision.io.read_image('./convolution/square.png').to(torch.float).to('cpu')
    image /= 255.

    cnn = CNN()
    cnn.to('cpu')

    print(cnn(image))

    square_filter = torch.zeros((3, 3, 10, 10))
    for channel in range(3):
        square_filter[channel, channel, :, :] = 1.

    for name, param in cnn.named_parameters():
        param.data = square_filter

    detection = cnn(image)
    print(detection)
    print(detection.shape)

    detection[1, :, :] = 0
    detection[2, :, :] = 0
    non_zero = torch.nonzero(detection)
    xvalues = set([x.item() for _, x, _ in non_zero])
    yvalues = set([y.item() for _, _, y in non_zero])
    
    for x in xvalues:
        detection[0, x, :] = 255
    for y in yvalues:
        detection[0, :, y] = 255

    pyplot.clf()
    pyplot.imshow(torch.permute(detection, (1, 2, 0)).detach().numpy())
    pyplot.show()