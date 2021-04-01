# This is a _very simple_ example of a web service that recognizes faces in uploaded images.
# Upload an image file and it will check if the image contains a picture of Barack Obama.
# The result is returned as json. For example:
# 通过命令行来post文件
# $ curl -XPOST -F "file=@obama2.jpg" http://127.0.0.1:5001/upload_image
#

# python代码访问该flask服务。通过request来post “multipart/form-data”
# import os, requests
# url = "http://localhost:5002/upload_image"
# file_path = '/home/WuPuQu/face_recognition-docker/examples/two_people.jpg'
# # file_path = '/home/WuPuQu/face_recognition-docker/examples/biden.jpg'
# file_name = file_path.split('/')[-1]
# print('file_name:',file_name)
# files={'file':(file_name, open(file_path,'rb'))}
# r = requests.post(url,files=files)
# print(r.text)

# This example is based on the Flask file upload example: http://flask.pocoo.org/docs/0.12/patterns/fileuploads/

# NOTE: This example requires flask to be installed! You can install it with pip:
# $ pip3 install flask

import face_recognition
from flask import Flask, jsonify, request, redirect
import numpy as np
import os
import zipfile

# You can change this to any folder on your system
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
ALLOWED_EXTENSIONS_2 = {'zip', 'gz', 'tar'}

app = Flask(__name__)


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def allowed_file_2(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS_2


@app.route('/upload_image', methods=['GET', 'POST'])
def upload_image():
    # Check if a valid image file was uploaded
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)

        file = request.files['file']

        if file.filename == '':
            return redirect(request.url)

        if file and allowed_file(file.filename):
            # The image file seems valid! Detect faces and return the result.
            return detect_faces_in_image(file)

    # If no valid image file was uploaded, show the file upload form:
    return '''
    <!doctype html>
    <title>who is this a picture ?</title>
    <h1>Upload a picture and see if it's a picture of who</h1>
    <form method="POST" enctype="multipart/form-data">
      <input type="file" name="file">
      <input type="submit" value="Upload">
    </form>
    '''


def detect_faces_in_image(file_stream):
    # Load the uploaded image file
    image = face_recognition.load_image_file(file_stream)
    # 检测人脸
    try:
        face_locations = face_recognition.face_locations(image)
    except:
        face_locations = []
    if len(face_locations) == 0: # 检测不到人脸
        return {}

    # Pre-calculated face encoding
    facebank_path = './data/facebank'
    targets, names = load_facebank(facebank_path)

    # Get face encodings for any faces in the uploaded image
    tolerance = 0.6
    pred_dict = {}
    for i in range(len(face_locations)):
        unknown_face_encoding = face_recognition.face_encodings(image)[i]
        face_distances = face_recognition.face_distance(targets, unknown_face_encoding)
        min_idx = np.where(face_distances == min(face_distances))
        # print(face_distances)
        # print(min_idx[0][0])
        # print('{:.3f}'.format(face_distances[min_idx][0]))
        if face_distances[min_idx][0] < tolerance:
            # print(names[min_idx[0][0] + 1], '{:.3f}'.format(face_distances[min_idx][0]))
            pred_dict[names[min_idx[0][0] + 1]] = round(face_distances[min_idx][0],3)
    if len(pred_dict) == 0:
        pred_dict[names[0]] = 1
    # print(pred_dict)
    return pred_dict



def load_facebank(facebank_path): # 需要修改cuda tensor 为numpy
    embeddings = np.load(facebank_path+'/facebank.npy')
    names = np.load(facebank_path+'/names.npy')
    return embeddings, names



@app.route('/prepare_facebank', methods=['GET', 'POST'])
def upload_faces():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)

        file = request.files['file']

        if file.filename == '':
            return redirect(request.url)

        if file and allowed_file_2(file.filename):
            # 解压文件夹
            # 人脸库zip压缩文件，文件格式
            # /name1/xxx1.jpg
            # /name2/xxx1.jpg
            zfile = zipfile.ZipFile(file, "r")
            zfile.extractall()
            # 处理文件
            facebank_path = './'+ file.filename.rsplit('.', 1)[0]  # './data/facebank' # 获取当前文件夹
            try:
                targets, names = prepare_facebank(facebank_path)
                result = {'upload_faces success': len(targets)}
            except:
                result = {'upload_faces success': 0}
            return result
    return '''
    <<!doctype html>
    <title>Upload new zipFile for facebank</title>
    <h1>Upload new zipFile for facebank</h1>
    <form action="" method=post enctype=multipart/form-data>
      <p><input type=file name=file>
         <input type=submit value=Upload>
    </form>
    '''


def prepare_facebank(facebank_path): # 待修改
    embeddings =  []
    names = ['Unknown']
    for path in os.listdir(facebank_path):
        if os.path.isdir(facebank_path+'/'+path):  # 判断是否是文件夹，不是文件夹才打开
            for path_ in os.listdir(facebank_path+'/'+path):
                try:
                    image = face_recognition.load_image_file(facebank_path+'/'+path+'/'+path_)
                    face_encoding = face_recognition.face_encodings(image)[0] # 单人脸照片
                    # print('face_encoding: \n',face_encoding)
                    embeddings.append(face_encoding)
                    names.append(path)
                except:
                    continue

    embeddings = np.array(embeddings)
    names = np.array(names)
    np.save('./data/facebank'+'/facebank', embeddings)
    np.save('./data/facebank'+'/names', names)
    return embeddings, names




if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5002, debug=True)
