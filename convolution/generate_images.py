from PIL import Image
import numpy
import random

XSIZE = 10
YSIZE = 10

if __name__ == '__main__':
    img = numpy.random.randint(low=0, high=256, size=(1000, 1000, 3))

    pic = Image.fromarray(img.astype('uint8'), 'RGB')
    pic.save('./convolution/random_noise.png')

    zeros = numpy.zeros((1000, 1000, 3))
    pic = Image.fromarray(zeros.astype('uint8'), 'RGB')
    pic.save('./convolution/black.png')

    ones = 255 * numpy.ones((1000, 1000, 3))
    pic = Image.fromarray(ones.astype('uint8'), 'RGB')
    pic.save('./convolution/white.png')

    for _ in range(1):

        x_center = random.randrange(0, 900)
        y_center = random.randrange(0, 900)

        myimg = numpy.zeros((1000, 1000, 3))

        for x_pixel in range(1000):
            for y_pixel in range(1000):
                for channel in range(3):
                    if x_center < x_pixel < x_center + XSIZE and y_center < y_pixel < y_center + YSIZE:
                        myimg[x_pixel][y_pixel][channel] = 255

        mypic = Image.fromarray(myimg.astype('uint8'), 'RGB')
        mypic.save('./convolution/square.png')        
