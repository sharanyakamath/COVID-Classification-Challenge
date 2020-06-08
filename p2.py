from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.models import load_model

from tensorflow.keras import preprocessing
from tensorflow.keras import backend as K
from tensorflow.keras import models

import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt

image_size = 224

image_path = "covid.jpeg"
model_path = "model2.h5"

# Load pre-trained Keras model and the image to classify
model = load_model(model_path) 
image = load_img(image_path)

# image = np.random.random((image_size, image_size, 3))
img_tensor = preprocessing.image.img_to_array(image)
img_tensor = np.expand_dims(img_tensor, axis=0)
# img_tensor = preprocess_input(img_tensor)

conv_layer = model.get_layer("conv2d_3")
heatmap_model = models.Model([model.inputs], [conv_layer.output, model.output])

# Get gradient of the winner class w.r.t. the output of the (last) conv. layer
with tf.GradientTape() as gtape:
    conv_output, predictions = heatmap_model(img_tensor)
    loss = predictions[:, np.argmax(predictions[0])]
    grads = gtape.gradient(loss, conv_output)
    pooled_grads = K.mean(grads, axis=(0, 1, 2))

heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_output), axis=-1)
heatmap = np.maximum(heatmap, 0)
max_heat = np.max(heatmap)
if max_heat == 0:
    max_heat = 1e-10
heatmap /= max_heat

print(heatmap.shape)
# plt.matshow(heatmap)
# plt.show()
# print(heatmap)

# subprot_args = {
#     'nrows': 1,
#     'ncols': 1,
#     'figsize': (6, 3),
#     'subplot_kw': {'xticks': [], 'yticks': []}
# }

# f, ax = plt.subplots(**subprot_args)
# ax.imshow(image)
# ax.imshow(heatmap, cmap='jet', alpha=0.5)
# # for i in range(len(cam)):
# #     heatmap = np.uint8(cm.jet(cam[i])[..., :3] * 255)
# #     ax[i].imshow(images[i])
# #     ax[i].imshow(heatmap, cmap='jet', alpha=0.5)
# plt.tight_layout()
# plt.show()
