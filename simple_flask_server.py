# !/usr/bin/env python
# encoding: utf-8
import json
from flask import Flask, request, jsonify
from flask_cors import CORS,cross_origin
import pickle

app = Flask(__name__)
cors=CORS(app)

loaded_model = pickle.load(open('finalized_model.sav', 'rb'))

@app.route('/', methods=['POST'])
@cross_origin()
def get_prediction():
    req_json = json.loads(request.data) #read request
    x = [req_json['data']]
    result = loaded_model.predict(x)
    print(result)
    result_float = [i for i in result]
    return jsonify(result_float) # return model output
    
app.run(debug=True)