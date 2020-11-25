from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
from flask import Flask, redirect, url_for, request, render_template
from flask_wtf import FlaskForm
from wtforms import SubmitField,IntegerField
from flask_bootstrap import Bootstrap
from werkzeug.utils import secure_filename
import os
from keras.applications.vgg16 import VGG16
from tensorflow.keras.models import model_from_json



app=Flask(__name__)
#Bootstrap(app)
app.config['SECRET_KEY']='srikar'
UPLOAD_FOLDER='static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

labels=['COVID19', 'NORMAL', 'PNEUMONIA']


def load_image(img_path):
    img=image.load_img(img_path,target_size=(200,200))
    img=image.img_to_array(img)
    img=np.expand_dims(img,axis=0)
    img=preprocess_input(img)
    return img

def prediction(model,img_path):
	class_type=np.argmax(model.predict(load_image(img_path)),axis=1)
	return labels[class_type[0]]

class model_form(FlaskForm):
	submit=SubmitField("Predict")



@app.route('/',methods=['GET','POST'])
def index():
	global img_path
	form=model_form()
	if request.method == 'POST':
		file=request.files['file']
		filename = secure_filename(file.filename)
		img_path =os.path.join(app.config['UPLOAD_FOLDER'], filename)
		file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
		return redirect(url_for('predict'))
	
	return render_template('index.html',form=form)

@app.route('/predict')
def predict():
	json_file=open("model.json","r")
	loaded_model_json=json_file.read()
	json_file.close()
	loaded_model=model_from_json(loaded_model_json)
	loaded_model.load_weights("model.h5")
	loaded_model.compile(loss="categorical_crossentropy",
		                 optimizer="adam",metrics=["accuracy"])
	result=prediction(loaded_model,img_path)
	return render_template('predict.html',result=result,img_path=img_path)



if __name__ == '__main__':
	app.run(debug=True)




