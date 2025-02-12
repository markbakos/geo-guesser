import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
from typing import Tuple
from keras import layers, models, preprocessing, applications

class AttentionVisualizer:
    def __init__(self, model, last_conv_layer_name='top_conv'):
        self.model = model
        self.last_conv_layer = self._get_last_conv_layer(last_conv_layer_name)
        self.grad_model = models.Model(
            [self.model.inputs],
            [self.last_conv_layer.output, self.model.output]
        )

    def _get_last_conv_layer(self, layer_name):
        for layer in reversed(self.model.layers):
            if isinstance(layer, layers.Conv2D) and layer_name == layer_name:
                return layer
        raise ValueError(f"Conv layer {layer_name} not found in model")

    def _compute_heatmap(self, img_array, pred_index=None):
        with tf.GradientTape() as tape:
            conv_outputs, predictions = self.grad_model(img_array)
            if pred_index is None:
                pred_index = tf.argmax(predictions[0])
            loss = predictions[0][:, pred_index]

        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))

        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        return heatmap.numpy()

    def overlay_heatmap(self, img_array, heatmap, alpha=0.4):
        heatmap = cv2.resize(heatmap, (img_array.shape[1], img_array.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        superimposed_img = heatmap * alpha + img_array
        superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)
        return superimposed_img

    def visualize(self, img_path: str, target_size: Tuple[int, int] = (224, 224)):
        img = preprocessing.image.load_img(img_path, target_size=target_size)
        img_array = preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, axis=0)
        img_array = applications.efficientnet.preprocess_input(img_array)

        heatmap = self._compute_heatmap(img_array)
        original_img = cv2.imread(img_path)
        original_img = cv2.resize(original_img, target_size)
        overlay = self.overlay_heatmap(original_img, heatmap)

        plt.figure(figsize=(10, 5))
        plt.subplot(1,2,1)
        plt.imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
        plt.title('Original Image')

        plt.subplot(1, 2, 2)
        plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        plt.title('Attention Heaatmap')
        plt.show()
        return overlay