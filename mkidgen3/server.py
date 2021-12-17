from logging import getLogger
from flask import Flask, request
from flask_restful import Api, Resource, reqparse
from .util import setup_logging
import mkidgen3 as gen3
import numpy as np

SERVER_PORT = 50001


class IQCaptureAPI(Resource):
    def get(self, n_samples):
        try:
            data = gen3.iqcapture(int(n_samples))
        except RuntimeError:
            return 'Error: did you set frequencies?', 400
        return {'data': data}, 200


class IQConfigAPI(Resource):
    def post(self):
        try:
            print('setting freq')
            gen3.set_frequencies(request.json['frequencies'], request.json.get('amplitudes',None))
        except ValueError:
            getLogger(__name__).info(f'Bad frequencies: {request.json}')
            return 'Bad JSON data', 400

setup_logging('gen3-flask')
app = Flask(__name__, static_url_path="")
api = Api(app)
api.add_resource(IQConfigAPI, '/set_freq')
api.add_resource(IQCaptureAPI, '/iqcap/<n_samples>')
# app.run(host='0.0.0.0', debug=False, port=SERVER_PORT)
