from PIL import Image
import numpy as np
from .process import process_image

def predict(image_path: str, model, top_k=1):
    ''' Predict the top k classes of an image

        params:
            image_path - a path to an image
            model - a keras model saved as an .h5
            top_k - the number of top classes
    '''
    image = Image.open(image_path)
    image = np.asarray(image)
    image = np.expand_dims(process_image(image), 0)

    predictions = model.predict(image)[0]

    probabilities = np.sort(predictions)[-top_k:len(predictions)]
    probabilities =  probabilities.tolist()
    ''' partition the array against the top_kth probability and return
        the indices (also the classes) of the top_k probabilities.

        convert top_classes to a list for plotting purposes.

        shift each class by 1 and convert to string
        in order to obtain the names from the .json that
        maps classes to flower names.

        lastly, return a tuple containing the probabilities and
        the top k classes.
    '''
    top_classes = np.argpartition(predictions, -top_k)[-top_k:]
    top_classes = top_classes.tolist()
    top_classes = [str(x + 1) for x in top_classes]
    return probabilities, top_classes
