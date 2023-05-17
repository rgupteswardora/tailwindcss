from flask import Flask,render_template
app = Flask(__name__)
from flask import Flask, render_template
import os
from flask import Flask, request, render_template_string, send_file, render_template, send_file, url_for
import io
app = Flask(__name__)
import os.path
import tensorflow as tf
from tensorflow.keras.models import load_model
from config import myConfig as config
import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import argparse
from pathlib import Path
import tensorflow.keras.backend as K
from matplotlib import pyplot as plt
import scipy.io
from scipy import ndimage
import hdf5storage
from tifffile import imwrite
from PIL import Image
import skimage.feature
from flask_ngrok import run_with_ngrok
app = Flask(__name__)
# custom filter
def my_Hfilter(shape, dtype=None):

    f = np.array([
            [[[-1]], [[0]], [[1]]],
            [[[-2]], [[0]], [[2]]],
            [[[-1]], [[0]], [[1]]]
        ])
    assert f.shape == shape
    return K.variable(f, dtype='float32')
    
def my_Vfilter(shape, dtype=None):

    f = np.array([
            [[[-1]], [[-2]], [[-1]]],
            [[[0]], [[0]], [[0]]],
            [[[1]], [[2]], [[1]]]
        ])
    assert f.shape == shape
    return K.variable(f, dtype='float32')

#ParsingArguments
parser=argparse.ArgumentParser()
#parser.add_argument('--dataPath',dest='dataPath',type=str,default='./Testing_data/BSD68/',help='testDataPath')
parser.add_argument('--weightsPath',dest='weightsPath',type=str,default='/Users/gupteswar/Downloads/B7.h5',help='pathOfTrainedCNN')
args=parser.parse_args()
#createModel, loadWeights
def custom_loss(y_true,y_pred): #this is required for loading a keras-model created with custom-loss
    diff=y_true-y_pred
    res=K.sum(diff*diff)/(2*config.batch_size)
    return res

model=load_model(args.weightsPath,custom_objects={'my_Hfilter': my_Hfilter,'my_Vfilter': my_Vfilter,'custom_loss':custom_loss})
print('Trained Model is loaded')
# Load the Keras model

# HTML template for displaying the output image
template = '''
<!doctype html>
<html>
    <body>
        <h1>Prediction result:</h1>
        <img src="data:image/png;base64,{{img_data}}" />
    </body>
</html>
'''

@app.route('/')
def home():
    return render_template("index.html")
@app.route('/product')
def product():
    return render_template('product.html')
@app.route('/team')
def team():
    return render_template('team.html')
@app.route('/button1', methods=['GET', 'POST'])
def button1():
    # code to execute when button1 is clicked
    if request.method == 'POST':
        file = request.files['file']
        if file:
            # Read the image file in memory
            img = Image.open(io.BytesIO(file.read()))

            # Preprocess the image
            arr = np.asarray(img)/ 255.0
            error=model.predict(np.expand_dims(arr,axis=0))
            predClean=arr-np.squeeze(error)
            print(arr)
            print(error)
            z=(predClean)
            cv2.imwrite("./static/"+"image.png",255.*z)
            # Make a prediction using the model
            # Convert the output image to PNG format
            return render_template('display_image.html')
    return '''
    <!doctype html>
    <html>
        <body>
            <h1>Upload an image</h1>
            <form method="POST" enctype="multipart/form-data">
                <input type="file" name="file">
                <input type="submit" value="Upload">
            </form>
        </body>
    </html>
    '''
    return 'Button 1 clicked'
@app.route('/button2', methods=['GET', 'POST'])
def button2():
    # code to execute when button2 is clicked
     return render_template('BUTTON2.html')
@app.route('/', methods=['POST'])
def upload_image():
    # Get the uploaded image
    image = request.files['image'].read()

    # Convert the image to a NumPy array
    npimg = np.frombuffer(image, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_GRAYSCALE)
    #img = tf.image.rgb_to_grayscale(img1)
    # Get the noise level from the user
    noise_level = float(request.form['noise_level'])

    # Add noise to the image
    #noise = np.zeros(img.shape, np.uint8)
    #cv2.randn(noise, np.zeros(3), np.ones(3) * 255 * noise_level)
    arr = np.asarray(img)/ 255.0
    f = arr + np.random.normal(0, noise_level/255., arr.shape)
    
    #clean image using our model
    error=model.predict(np.expand_dims(f,axis=0))
    predClean=f-np.squeeze(error)
    z=(predClean)
    # Save the noisy image to disk
    cv2.imwrite('./static/noisy_image.png', 255.*f)
    cv2.imwrite("./static/"+"final_image.png",255.*z)
    return render_template('show_images.html', noisy_img='static/noisy_image.jpg')
app.run(debug=True)
"""
This package has installed:
	•	Node.js v18.16.0 to /usr/local/bin/node
	•	npm v9.5.1 to /usr/local/bin/npm
Make sure that /usr/local/bin is in your $PATH"""
