from flask import Flask, request, jsonify, render_template
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from werkzeug.utils import secure_filename
import traceback
import cv2
import time

app = Flask(__name__, static_folder='static', template_folder='.')
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('static', exist_ok=True)

# We will load the image model instead of the SVC model
MODEL_PATH = 'stress_image_model.h5'
print(f"Loading Image Model from {MODEL_PATH}...")
try:
    model = tf.keras.models.load_model(MODEL_PATH)
except Exception as e:
    print(f"Model not found yet. It will be loaded after training. {e}")
    model = None

def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    last_conv_layer = model.get_layer(last_conv_layer_name)
    last_conv_layer_model = tf.keras.Model(model.inputs, last_conv_layer.output)

    classifier_input = tf.keras.Input(shape=last_conv_layer.output.shape[1:])
    x = classifier_input
    
    layer_names = [layer.name for layer in model.layers]
    start_idx = layer_names.index(last_conv_layer_name) + 1
    
    for layer in model.layers[start_idx:]:
        x = layer(x)
        
    classifier_model = tf.keras.Model(classifier_input, x)

    with tf.GradientTape() as tape:
        last_conv_layer_output = last_conv_layer_model(img_array)
        tape.watch(last_conv_layer_output)
        preds = classifier_model(last_conv_layer_output)
        class_channel = preds[:, 0]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    if grads is None:
        return np.zeros((img_array.shape[1], img_array.shape[2]))
        
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
    return heatmap.numpy()

def save_gradcam(img_path, heatmap, cam_path, alpha=0.4):
    img = cv2.imread(img_path)
    if img is None: return False
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = heatmap * alpha + img
    cv2.imwrite(cam_path, superimposed_img)
    return True

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_endpoint():
    global model
    if model is None:
        try:
            model = tf.keras.models.load_model(MODEL_PATH)
        except:
            return jsonify({'error': 'Model is still training. Please wait.'}), 503
            
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
        
    if file and file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Parse and predict image
            img = load_img(filepath, target_size=(224, 224))
            img_array = img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            # Predict
            pred_prob = float(model.predict(img_array)[0][0])
            is_stressed = pred_prob > 0.5
            
            # Generate Grad-CAM Heatmap
            # The custom CNN we built has layers like conv2d, conv2d_1, conv2d_2. 
            # We use the last conv layer for Grad-CAM.
            last_conv_layer = "conv2d_2" 
            heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer)
            
            heatmap_filename = f"cam_{int(time.time())}.jpg"
            heatmap_path = os.path.join('static', heatmap_filename)
            save_gradcam(filepath, heatmap, heatmap_path)
            
            # Clean up original upload
            os.remove(filepath)
            
            return jsonify({
                'success': True,
                'filename': filename,
                'probability': pred_prob,
                'is_stressed': is_stressed,
                'message': 'STRESSED (มีความเครียด)' if is_stressed else 'NORMAL (ปกติ)',
                'heatmap_url': f'/static/{heatmap_filename}'
            })
        except Exception as e:
            traceback.print_exc()
            return jsonify({'error': f'Error processing image: {str(e)}'}), 500
            
    return jsonify({'error': 'Invalid file format. Please upload .jpg or .png file'}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)
