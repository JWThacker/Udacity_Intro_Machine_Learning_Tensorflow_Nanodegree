import warnings
warnings.filterwarnings('ignore')
import json
import sys
import argparse as ap

import numpy as np

import tensorflow as tf
tf.get_logger().setLevel('WARNING')
tf.autograph.set_verbosity(2)
import tensorflow_hub as hub
import logging

from utils import process_image, predict
'''
   Predict the flower species given an image of a flower.

   params:
       /path/to/image - a path to an image to make a prediction from.
       saved_model - a Keras model saved as a .h5
       --top_k - the top number of classes that image could be.
       --category_names - path to a .json labeling classes to species names.
'''
def main():

    # Add and then parse all command line arguments.
    parser = ap.ArgumentParser(usage=('python3 predict.py /path/to/image saved_model '
                                      '--top_k K --category_names map.json'),
                               description=('Predict the species'
                                            ' of a flower image.'))

    parser.add_argument('image_path', type=str,  help='Path to an image of a flower')

    parser.add_argument('saved_model', type=str,  help='A tf.Keras model saved as an .h5')

    parser.add_argument('--top_k', type=int, default=1, help=('Number of different'
                                                              ' species probabilities'
                                                              ' will be displayed for'))

    parser.add_argument('--category_names', type=str,
                                      default=None,
                                      help=('path to a .json file'
                                            'containing the mapped'
                                            'names of the predicted'
                                            'species of flowers'))
    args = parser.parse_args()

    # Load saved Keras model
    reloaded_model = tf.keras.models.load_model(args.saved_model, custom_objects={'KerasLayer': hub.KerasLayer})

    # predict the species with the corresponding probabilities
    try:
        probs, classes = predict(args.image_path, reloaded_model, args.top_k)
    except FileNotFoundError:
        print('\n\n')
        print('Image not found; enter a valid path to an image')
        print('\n\n')
        sys.exit()
    else:
        # If --category_names was not empty, map class labels to species names
        if args.category_names:
            species_names = []
            try:
                with open(args.category_names, 'r') as f:
                    class_names = json.load(f)
            except FileNotFoundError:
                print('\n\n')
                print(f'{args.category_names} not found; enter valid path.')
                print('\n\n')
                sys.exit()
            else:
                for i, classs in enumerate(classes):
                    species_names.append(class_names[classs])
                results = {name: prob for name, prob in zip(species_names, probs)}
                print('\n\n')
                print('Flower Species Name: Probability of species')
                for name in species_names:
                    print(name.title(), ': ', results[name])
                print('\n\n')
        # Otherwise print the class labels and corresponding probabilities
        else:
            print('\n\n')
            results = {classs: prob for classs, prob in zip(classes, probs)}
            print('Class Label: Probability of class')
            for classs in classes:
                print(classs, ': ', results[classs])
            print('\n\n')

if __name__ == '__main__':
    main()
