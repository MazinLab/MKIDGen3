from logging import getLogger
from flask import Flask, request, Response, stream_with_context
from flask_restful import Api, Resource, reqparse
from .util import setup_logging
import mkidgen3 as gen3
import numpy as np
import blosc

SERVER_PORT = 50001


def compress_array_chunks(data):
    for index in np.arange(data.shape[0]):
        yield blosc.compress_ptr(data[index, ...].__array_interface__['data'][0], data.size,
                                 data.dtype.itemsize, 9, True)


def serve_compressed_array(a, name):
    return Response(
        stream_with_context(compress_array_chunks(a)),
        headers={
            'Content-Disposition': f'attachment; filename={name}'
        }
    )


class IQCaptureAPI(Resource):
    def get(self, n_samples):
        try:
            n_samples=int(n_samples)
            print(f'caturing {n_samples}')
            getLogger(__name__).info(f'caturing {n_samples}')
            # data = list(np.arange(n_samples)/n_samples)
            data = gen3.iqcapture(int(n_samples))
            return serve_compressed_array(data, 'foo.np')
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

import requests, tempfile
def get_data():
    def fetch_blosc_array(url):
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with tempfile.TemporaryFile(mode='w+b') as tf:
                for chunk in r.iter_content(chunk_size=8192):
                    tf.write(chunk)
                tf.seek(0)
                array = blosc.unpack_array(tf)
        return array

    return fetch_blosc_array('{url}/capture/{source}/{nsamples}')
