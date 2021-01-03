import os 
from flask import Flask 
from flask import render_template
from flask import request
from flask import redirect
from flask import url_for
import utils
import torch

app = Flask(__name__, template_folder='templates')
UPLOAD_FOLDER = 'C:/Users/acer/Desktop/Tutorial Web App/static/img'

@app.route("/", methods=['GET', 'POST'])
def upload_predict():
	if request.method == 'POST':
		image_file = request.files['image']
		if image_file:
			image_location = os.path.join(
				UPLOAD_FOLDER, image_file.filename
				)

			image_file.save(image_location)
			result = utils.predict(image_location, MODEL, MAPPING)
			
			return render_template("index.html", 
				prediction=result['dog'], 
				filename=image_file.filename)

	return render_template("home.html")

@app.route('/display/<filename>')
def display_image(filename):
	return redirect(url_for('static', filename='img/' + filename), code=301)

if __name__ == '__main__':
	checkpoint = torch.load('./results/checkpoint.pth', map_location='cpu')
	n_classes = checkpoint['n_classes']
	MAPPING = checkpoint['inv_mapping_label']
	MODEL = utils.Net(n_classes=n_classes, pretrained=True)	#.to(device=utils.DEVICE)
	MODEL.load_state_dict(checkpoint['state_dict'])
	MODEL.eval()
	app.run(port=int(os.environ.get('PORT', 5000)), debug=True, extra_files=UPLOAD_FOLDER)