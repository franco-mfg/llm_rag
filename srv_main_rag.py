# https://blog.stoplight.io/python-rest-api

import os
import sys

sys.path.append('../')

import pytools.utils as tls

pkg_list=[
  'requests',
  'flask',
  'langchain_community',
  'langchain_chroma',
  # 'streamlit'
]

if tls.pip_install(pkg_list)>0:
  tls.pip_save_requirements(pkg_list)

import flask
from flask import request, jsonify, send_file
from rag import do_query

app = flask.Flask(__name__)

@app.route('/')
def home():
    print(f'getting...file')
    return send_file('index.html', mimetype='text/html')

@app.route('/api', methods=['GET','POST'])
def query():
    """
        Gestisce richieste GET e POST all'endpoint /query.
    """
    query_params = request.args.to_dict()
    if request.method == 'GET':
        # query_params = request.args.to_dict()
        query=request.args.get("query",None)
        multi=request.args.get("multi",None)
        print(f'q={query}')

        if(query!=None):
          response=do_query(query,multi in ['si','yes','on','1'])
        else:
          response={'errore':'manca query'}

        print(response)
        return jsonify(response)

    elif request.method == 'POST':
        data = request.get_json()
        return jsonify(data)
    else:
        return jsonify({'error': 'Metodo HTTP non supportato'}), 405


@app.route('/info', methods=['GET'])
def info():
    """Handles GET requests to the /info endpoint.

    Returns:
        JSON response containing information about the server.
    """
    info = {
        'server_name': 'My Flask Server',
        'version': '1.0.0',
        'description': 'A simple API server for querying and retrieving information.',
        # 'db': vectorstore.get(limit=1)
    }
    return jsonify(info)

if __name__ == '__main__':
    app.run(debug=True, port=5080)