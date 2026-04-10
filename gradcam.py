import cv2
import numpy as np
import tensorflow as tf
import os
import uuid

OUTPUT_FOLDER = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def generate_gradcam(model, img_path):
    # Load image
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Failed to load image from {img_path}")

    img_resized = cv2.resize(img, (384, 384))
    img_resized = img_resized.astype("float32") / 255.0
    input_img = np.expand_dims(img_resized, axis=0)

    # Find last conv layer
    last_conv_layer = None
    try:
        last_conv_layer = model.get_layer("top_conv")
    except:
        for layer in reversed(model.layers):
            try:
                if len(layer.output.shape) == 4:
                    last_conv_layer = layer
                    break
            except:
                continue
    if last_conv_layer is None:
        raise ValueError("No convolution layer found for GradCAM")

    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[last_conv_layer.output, model.output]
    )

    # Compute gradients
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(input_img)
        if isinstance(predictions, list):
            predictions = predictions[0]

        class_idx = tf.argmax(predictions[0])
        loss = predictions[0, class_idx]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))
    conv_outputs = conv_outputs[0]

    # Compute heatmap
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap).numpy()

    # Ensure 2D float32 heatmap
    if heatmap.ndim != 2:
        heatmap = np.squeeze(heatmap)
    heatmap = np.maximum(heatmap, 0)
    heatmap = heatmap.astype(np.float32)

    if np.max(heatmap) > 0:
        heatmap /= np.max(heatmap)
    else:
        heatmap = np.zeros((img_resized.shape[0], img_resized.shape[1]), dtype=np.float32)

    # Resize to original image size safely
    try:
        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    except Exception:
        # fallback to model input size if resize fails
        heatmap = cv2.resize(heatmap, (384, 384))
        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Superimpose
    superimposed = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

    # Save
    filename = f"gradcam_{uuid.uuid4().hex}.png"
    output_path = os.path.join(OUTPUT_FOLDER, filename)
    if not cv2.imwrite(output_path, superimposed):
        raise RuntimeError(f"Failed to save GradCAM image at {output_path}")

    return output_path