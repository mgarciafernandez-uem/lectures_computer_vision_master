from PIL import Image
import numpy
import random

XSIZE = 10
YSIZE = 10

if __name__ == '__main__':
    img = numpy.random.randint(low=0, high=256, size=(1000, 1000, 3))

    channel_dict = {0: 'red', 1: 'green', 2: 'blue'}

    for _ in range(1000):

        x_center = random.randrange(0, 50-XSIZE)
        y_center = random.randrange(0, 50-XSIZE)

        for channel in range(3):

            myimg = numpy.zeros((50, 50, 3))

            for x_pixel in range(50):
                for y_pixel in range(50):

                    if x_center <= x_pixel < x_center + XSIZE and y_center <= y_pixel < y_center + YSIZE:
                        myimg[x_pixel][y_pixel][channel] = 255

            mypic = Image.fromarray(myimg.astype('uint8'), 'RGB')
            mypic.save(f'./convolution/images/{channel_dict[channel]}/square_{_}.png')    