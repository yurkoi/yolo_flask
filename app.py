import io
from PIL import Image
import cv2
import torch
from flask import Flask, render_template, request, make_response
# from flask_ngrok import run_with_ngrok
from werkzeug.exceptions import BadRequest
import os
import time


app = Flask(__name__)
# run_with_ngrok(app)
dictOfModels = {}
listOfKeys = []
#
#
# def check_pretrained_models():
#     # create a python dictionary for your models d = {<key>: <value>, <key>: <value>, ..., <key>: <value>}
#     # create a list of keys to use them in the select part of the html code
#     # global dictOfModels
#     # global listOfKeys
#
#


def get_prediction(img_bytes, model):
    img = Image.open(io.BytesIO(img_bytes))
    # inference
    results = model(img, size=840)
    return results


@app.route('/', methods=['POST'])
def predict():
    file = extract_img(request)
    img_bytes = file.read()
    # choice of the model
    results = get_prediction(img_bytes, dictOfModels[request.form.get("model_choice")])
    print(f'User selected model : {request.form.get("model_choice")}')
    # updates results.imgs with boxes and labels
    print(results)
    results.render()
    # encoding the resulting image and return it
    for img in results.imgs:
        RGB_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        im_arr = cv2.imencode('.jpg', RGB_img)[1]
        response = make_response(im_arr.tobytes())
        response.headers['Content-Type'] = 'image/jpeg'
    # return your image with boxes and labels
    return response


def extract_img(request):
    # checking if image uploaded is valid
    if 'file' not in request.files:
        raise BadRequest("Missing file parameter!")
    file = request.files['file']
    if file.filename == '':
        raise BadRequest("Given file is invalid")
    return file


@app.route('/', methods=['GET'])
def get():
  # in the select we will have each key of the list in option
  return render_template("index.html", len=len(listOfKeys), listOfKeys=listOfKeys)


if __name__ == '__main__':

    for r, d, f in os.walk("models_train"):
        for file in f:
            if ".pt" in file:
                # example: file = "model1.pt"
                # the path of each model: os.path.join(r, file)
                dictOfModels[os.path.splitext(file)[0]] = torch.hub.load('ultralytics/yolov5', 'custom',
                                                                         path=os.path.join(r, file), force_reload=True)
                # you would obtain: dictOfModels = {"model1" : model1 , etc}
        for key in dictOfModels:
            listOfKeys.append(key)  # put all the keys in the listOfKeys
        time.sleep(2)
        print(listOfKeys)

    app.run(debug=True, host='0.0.0.0')
    # app.run()
    # check_pretrained_models()

