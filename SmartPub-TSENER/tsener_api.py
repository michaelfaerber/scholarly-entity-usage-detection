import os
from flask import Flask, jsonify, request
from flasgger import Swagger
from flasgger.utils import swag_from
from m2_labelling import ner_labelling
from config import ROOTPATH
import flask

app = Flask(__name__)
template = {
    "swagger": "2.0",
    "info": {
      "description": "This is the server for users interested in training and using NER models to label long-tail "
                     "entities in text.  Find more information about TSE-NER in the [paper](to-do.tudelft) and check "
                     "out our [code](https://github.com/mvallet91/SmartPub-TSENER) for more details and if you want "
                     "to create your own application.",
      "version": "1.0.0",
      "title": "SmartPub TSE-NER",
      "termsOfService": "http://swagger.io/terms/",
      "contact": {
        "email": "m.valletorre@tudelft.nl"},
      "license": {
        "name": "GNU GPL v3.0",
        "url": "https://github.com/mvallet91/SmartPub-TSENER/blob/master/LICENSE"}
            }
        }
Swagger(app, template=template)


# @app.route('/api/<string:model_name>/', methods=['GET'])
# @swag_from('m1_index.yml')
# def m1(model_name):
#     word_list = model_name.split()
#     word_list = [w.lower().strip() for w in word_list]
#     model = str(request.args.get('model', 1))
#     results = filtering.filter_ws_fly(word_list)
#     return jsonify(
#         text=model_name,
#         model=model,
#         entities=results
#     )


@app.route('/api/<string:text>/', methods=['GET'])
@swag_from('m2_index.yml')
def m2(text):
    model = str(request.args.get('model', 1))
    path_to_model = ROOTPATH + '/crf_trained_files/' + model + '_TSE_model_3.ser.gz'
    if not os.path.isfile(path_to_model):
        flask.abort(501)
    if len(text) > 10000:
        flask.abort(502)
    results = ner_labelling.long_tail_labelling(model, text)
    return jsonify(
        text=text,
        model=model,
        entities=results
    )


app.run(debug=True)
