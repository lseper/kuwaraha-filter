from flask import Flask, request, send_file
from flask_cors import CORS
from kuwahara import library_kuwahara
from werkzeug.utils import secure_filename
import os

UPLOAD_FOLDER = './imgs'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'PNG', 'JPG', 'JPEG',])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
CORS(app, resources={r"*": {"origins": ["http://localhost:3000"]}})


@app.route('/filter', methods=['POST'])
def filter():
    print("==================== beginning request =========================")
    print(
        f"==================== Created target dir =========================")
    print(
        f"==================== request file =========================")
    file = request.files['file']
    print(file)
    print(f'kernel size: {request.form.get("kernel")}')
    kernel = request.form.get("kernel")
    print(file.filename)
    filename = secure_filename(file.filename)
    destination = f"./imgs/{filename}"
    print(f'Destination: {destination}')
    file.save(destination)
    cleaned_name = filename.split(".")[0]
    kuw_path = library_kuwahara(destination, 'lagrange', int(
        kernel), dir='api', img_name=cleaned_name)
    print(f'kuw path: {kuw_path} =========================================')
    return send_file(kuw_path)
    # file = request.files['file']
    # data = request.json
    # print(data)
    # return send_file(custom_kuwahara(file, method="lagrange", radius=data['radius']))


if __name__ == "__main__":
    app.run()
