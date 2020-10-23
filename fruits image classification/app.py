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

labels=['Apple Braeburn', 'Apple Crimson Snow', 'Apple Golden 1', 'Apple Golden 2', 'Apple Golden 3', 'Apple Granny Smith', 'Apple Pink Lady', 'Apple Red 1', 'Apple Red 2', 'Apple Red 3', 'Apple Red Delicious', 'Apple Red Yellow 1', 'Apple Red Yellow 2', 'Apricot', 'Avocado', 'Avocado ripe', 'Banana', 'Banana Lady Finger', 'Banana Red', 'Beetroot', 'Blueberry', 'Cactus fruit', 'Cantaloupe 1', 'Cantaloupe 2', 'Carambula', 'Cauliflower', 'Cherry 1', 'Cherry 2', 'Cherry Rainier', 'Cherry Wax Black', 'Cherry Wax Red', 'Cherry Wax Yellow', 'Chestnut', 'Clementine', 'Cocos', 'Corn', 'Corn Husk', 'Cucumber Ripe', 'Cucumber Ripe 2', 'Dates', 'Eggplant', 'Fig', 'Ginger Root', 'Granadilla', 'Grape Blue', 'Grape Pink', 'Grape White', 'Grape White 2', 'Grape White 3', 'Grape White 4', 'Grapefruit Pink', 'Grapefruit White', 'Guava', 'Hazelnut', 'Huckleberry', 'Kaki', 'Kiwi', 'Kohlrabi', 'Kumquats', 'Lemon', 'Lemon Meyer', 'Limes', 'Lychee', 'Mandarine', 'Mango', 'Mango Red', 'Mangostan', 'Maracuja', 'Melon Piel de Sapo', 'Mulberry', 'Nectarine', 'Nectarine Flat', 'Nut Forest', 'Nut Pecan', 'Onion Red', 'Onion Red Peeled', 'Onion White', 'Orange', 'Papaya', 'Passion Fruit', 'Peach', 'Peach 2', 'Peach Flat', 'Pear', 'Pear 2', 'Pear Abate', 'Pear Forelle', 'Pear Kaiser', 'Pear Monster', 'Pear Red', 'Pear Stone', 'Pear Williams', 'Pepino', 'Pepper Green', 'Pepper Orange', 'Pepper Red', 'Pepper Yellow', 'Physalis', 'Physalis with Husk', 'Pineapple', 'Pineapple Mini', 'Pitahaya Red', 'Plum', 'Plum 2', 'Plum 3', 'Pomegranate', 'Pomelo Sweetie', 'Potato Red', 'Potato Red Washed', 'Potato Sweet', 'Potato White', 'Quince', 'Rambutan', 'Raspberry', 'Redcurrant', 'Salak', 'Strawberry', 'Strawberry Wedge', 'Tamarillo', 'Tangelo', 'Tomato 1', 'Tomato 2', 'Tomato 3', 'Tomato 4', 'Tomato Cherry Red', 'Tomato Heart', 'Tomato Maroon', 'Tomato Yellow', 'Tomato not Ripened', 'Walnut', 'Watermelon']


def load_image(img_path):
    img=image.load_img(img_path,target_size=(100,100))
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




