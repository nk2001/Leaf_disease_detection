from keras.preprocessing import image
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.resnet50 import preprocess_input
from keras.applications.resnet50 import decode_predictions
from PIL import Image
import numpy as np
from keras.models import load_model


def identify(image1): 
    model = load_model("model.h5")
    #Resize the image to 224x224
    image = load_img(image1, target_size=(224, 224))
    #convert the image pixels to a numpy array
    image = img_to_array(image)
    # reshape data for the model to add another dimension as keras wants the input as set of images which have 3D
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))

    # prepare the image for the model
    image = preprocess_input(image)
    # predict the probability across all output classes
    pred = model.predict(image)
    # convert the probabilities to the class labels
    label = np.argmax(pred, axis = 1)
	# retrieve the most likely result, e.g. highest probability
    label = label[0]
    if label:
	    label1 = "HEALTHY"
    else:
	    label1 = "NOT HEALTHY"
    return label1 