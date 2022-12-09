from logging import getLogger
from flask import Flask, request, Response, stream_with_context
from flask_restful import Api, Resource, reqparse

import mkidgen3.overlay_helpers
from mkidgen3.util import setup_logging
import mkidgen3 as gen3
import numpy as np
import blosc
import requests, tempfile

SERVER_PORT = 50001
CHUNK_SIZE = 8192


def compress_array_chunks(data):
    for index in np.arange(data.shape[0]):
        yield blosc.compress_ptr(data[index, ...].__array_interface__['data'][0], data.size,
                                 data.dtype.itemsize, 9, True)


def serve_compressed_array(a, name):
    return Response(blosc.pack_array(a),
                    headers={
                        'Content-Disposition': f'attachment; filename={name}'
                    }
                )
    # return Response(
    #     stream_with_context(compress_array_chunks(a)),
    #     headers={
    #         'Content-Disposition': f'attachment; filename={name}'
    #     }
    # )


class CaptureAPI(Resource):
    def get(self, tap, n_samples):
        try:
            n_samples=int(n_samples)
            print(f'caturing {n_samples}')
            getLogger(__name__).info(f'caturing {n_samples}')
            # data = list(np.arange(n_samples)/n_samples)
            if tap=='iq':
                data = gen3.iqcapture(int(n_samples))
            elif tap=='adc':
                pass
            else
            return serve_compressed_array(data, 'foo.np')
        except RuntimeError:
            return 'Error: did you set frequencies?', 400
        return {'data': data}, 200


class IQConfigAPI(Resource):
    def post(self):
        try:
            print('setting freq')
            mkidgen3.overlay_helpers.set_frequencies(request.json['frequencies'], request.json.get('amplitudes', None))
        except ValueError:
            getLogger(__name__).info(f'Bad frequencies: {request.json}')
            return 'Bad JSON data', 400


def remote_capture(tap, nsamples, url=f'http://mkidzcu111b.physics.ucsb.edu:{SERVER_PORT}'):
    def fetch_blosc_array(url):
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with tempfile.TemporaryFile(mode='w+b') as tf:
                for chunk in r.iter_content(chunk_size=CHUNK_SIZE):
                    tf.write(chunk)
                tf.seek(0)
                array = blosc.unpack_array(tf)
        return array

    return fetch_blosc_array(f'{url}/cap/{tap}/{nsamples}')


if __name__=='__main__':
    mkidgen3.overlay_helpers.configure(.....)
    setup_logging('gen3-flask')
    app = Flask(__name__, static_url_path="")
    api = Api(app)
    api.add_resource(IQConfigAPI, '/set_freq')
    api.add_resource(CaptureAPI, '/cap/<tap>/<n_samples>')
    # api.add_resource(IQCaptureAPI, '/adccap/<n_samples>')
    app.run(host='0.0.0.0', debug=False, port=SERVER_PORT)
