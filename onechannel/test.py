import tensorflow as tf
import numpy as np
from tensorflow.keras.applications.inception_v3 import InceptionV3

    

def grey_InceptionV3():
    base_model = InceptionV3(input_shape=(256,256,3))
    conv_kernel_weights = base_model.layers[1].get_weights()[0]
    converted_kernel_weights = np.sum(conv_kernel_weights, axis = 2)
    converted_kernel_weights = np.expand_dims(converted_kernel_weights, 2)
    
    new_model = InceptionV3(input_shape=(256,256,1), weights=None)
    new_model.layers[1].set_weights([converted_kernel_weights])
    for i in range(2, len(base_model.layers)):
        print(i)
        layer_weights = base_model.layers[i].get_weights()
        if len(layer_weights)>0:
            new_model.layers[i].set_weights(layer_weights)
    return new_model




img_grey = np.random.rand(1,256,256,1)
img_color = np.concatenate([img_grey,img_grey,img_grey], axis=-1)


model_original = InceptionV3(input_shape=(256,256,3))
model_modified = grey_InceptionV3()


