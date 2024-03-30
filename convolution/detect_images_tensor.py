from PIL import Image
import numpy
from matplotlib import pyplot

KXSIZE = 10
KYSIZE = 10
DETECTION_THRESHOLD = 245


if __name__ == '__main__':
    pic = Image.open("./convolution/square.png")
    image = numpy.array(pic)

    xsize, ysize, nchannels = image.shape

    r_image = numpy.array(image[:, :, 0])
    r_filter = numpy.ones((KXSIZE, KYSIZE))
    
    tensor_product = numpy.tensordot(r_filter, r_image, axes=0)

    distance_matrix = numpy.sum(tensor_product, axis=(0,1))
    distance_matrix /= KXSIZE * KYSIZE

    distance_matrix = numpy.array([distance_matrix, distance_matrix, distance_matrix])
    distance_matrix = numpy.transpose(distance_matrix, axes=(1, 2, 0))

    x_centers, y_centers = numpy.where(distance_matrix[:, :, 0] > DETECTION_THRESHOLD)

    pic = Image.fromarray(distance_matrix.astype('uint8'), 'RGB')
    pic.save('./convolution/tensor_product.png')

    for x_pixel in range(xsize):
        for y_pixel in range(ysize):
            if x_pixel in x_centers or y_pixel in y_centers:
                distance_matrix[x_pixel][y_pixel][0] = 255
                distance_matrix[x_pixel][y_pixel][1] = 0
                distance_matrix[x_pixel][y_pixel][2] = 0

                image[x_pixel][y_pixel][0] = 255
                image[x_pixel][y_pixel][1] = 0
                image[x_pixel][y_pixel][2] = 0

    pic = Image.fromarray(image.astype('uint8'), 'RGB')
    pic.save('./convolution/image_detected_tensor.png')