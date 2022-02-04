import numpy as np
import tensorflow as tf

image_size = 224

def process_image(image: np.array):
    ''' Process an image

        Process an image such that it is compatible
        with our neural network

        params:
            image - a path to an image
    '''
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, [image_size, image_size])
    image /= 255
    image = image.numpy().squeeze()
    return image
