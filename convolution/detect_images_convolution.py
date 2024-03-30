from PIL import Image
import numpy
from matplotlib import pyplot

KXSIZE = 10
KYSIZE = 10
DETECTION_THRESHOLD = 200


def kernel(image, k_x, k_y, k_c, k_xsize=KXSIZE, k_ysize=KYSIZE):

    convolution = numpy.sum(image[k_x:k_x + k_xsize, k_y:k_y + k_ysize, 0])
    convolution /= k_xsize * k_ysize

    return convolution  

if __name__ == '__main__':
    pic = Image.open("./convolution/square.png")
    image = numpy.array(pic)

    xsize, ysize, nchannels = image.shape

    distance_matrix = numpy.zeros((xsize, ysize, nchannels))
    for x_pixel in range(xsize-KXSIZE):
        for y_pixel in range(ysize-KYSIZE):
            dd = kernel(image=image, k_x=x_pixel, k_y=y_pixel, k_c=0)
            print(f'c: {0}, x: {x_pixel}, y: {y_pixel} -- {dd}')
            for channel in range(nchannels):
                distance_matrix[x_pixel + int(KXSIZE / 2)][y_pixel + int(KYSIZE / 2)][channel] = dd


    x_centers, y_centers = numpy.where(distance_matrix[:, :, 0] > DETECTION_THRESHOLD)

    for x_pixel in range(xsize):
        for y_pixel in range(ysize):
            if x_pixel in x_centers or y_pixel in y_centers:
                distance_matrix[x_pixel][y_pixel][0] = 255
                distance_matrix[x_pixel][y_pixel][1] = 0
                distance_matrix[x_pixel][y_pixel][2] = 0

                image[x_pixel][y_pixel][0] = 255
                image[x_pixel][y_pixel][1] = 0
                image[x_pixel][y_pixel][2] = 0

    pic = Image.fromarray(distance_matrix.astype('uint8'), 'RGB')
    pic.save('./convolution/convolution.png')

    pic = Image.fromarray(image.astype('uint8'), 'RGB')
    pic.save('./convolution/image_detected.png')