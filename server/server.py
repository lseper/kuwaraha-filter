from flask import Flask, request, send_file
from server.kuwahara import custom_kuwahara

app = Flask(__name__)

@app.route('/filter', methods=['POST'])
def filter():
  file = request.files['file']
  data = request.json

  return send_file(custom_kuwahara(file, method="lagrange", radius=data['radius']))