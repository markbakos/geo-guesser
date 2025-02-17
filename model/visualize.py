import numpy as np
import tensorflow as tf
import cv2
from typing import Tuple
from keras import models, preprocessing, applications

class AttentionVisualizer:
    """A class for visualizing model attention through Grad-CAM"""
    def __init__(self, model, last_conv_layer_name='top_conv'):
        self.model = model
        self.last_conv_layer = model.get_layer(last_conv_layer_name)
        self.grad_model = self._create_grad_model()

    def _create_grad_model(self):
        """Create a grad model for computing GRAD-CAM."""
        return models.Model(
            inputs=[self.model.inputs],
            outputs=[
                self.last_conv_layer.output,
                self.model.get_layer('city').output,
                self.model.get_layer('coordinates').output
            ]
        )

    def _compute_heatmap(self, img_array: np.ndarray) -> np.ndarray:
        """Compute the GRAD-CAM heatmap for an input image."""
        with tf.GradientTape() as tape:
            conv_output, city_preds, coord_preds = self.grad_model(img_array)
            pred_index = tf.argmax(city_preds[0])
            class_channel = city_preds[:, pred_index]

        grads = tape.gradient(class_channel, conv_output)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        conv_output = conv_output[0]
        heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_output), axis=-1)
        heatmap = tf.maximum(heatmap, 0) / tf.reduce_max(heatmap)

        return heatmap.numpy()

    def overlay_heatmap(self, img: np.ndarray, heatmap: np.ndarray, alpha: float = 0.4) -> np.ndarray:
        """Overlay the heatmap on the original image."""
        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        if len(img.shape) == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        superimposed = cv2.addWeighted(img, 1 - alpha, heatmap, alpha, 0)
        return cv2.cvtColor(superimposed, cv2.COLOR_BGR2RGB)

    def visualize(self, image_path: str, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
        """Generate a full visualization of model attention for an image"""
        img = preprocessing.image.load_img(image_path, target_size=target_size)
        img_array = preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = applications.efficientnet.preprocess_input(img_array)

        heatmap = self._compute_heatmap(img_array)

        original_img = cv2.imread(image_path)
        original_img = cv2.resize(original_img, target_size)

        overlay = self.overlay_heatmap(original_img, heatmap)

        return overlay