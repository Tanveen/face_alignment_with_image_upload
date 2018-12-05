import os
#from align_dlib import AlignDlib
from flask import Flask, request, render_template, send_from_directory
import dlib
import imutils
import cv2
import numpy as np
from PIL import Image
#from face_utils import facealigner
from imutils.face_utils import rect_to_bb
from imutils.face_utils import FaceAligner
import glob

app = Flask(__name__)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))


@app.route("/")
def index():
    return render_template("upload.html")


@app.route("/upload", methods=["POST"])
def upload():
    
    target = os.path.join(APP_ROOT, './output/uploads')
    print(target)
    if not os.path.isdir(target):
        os.mkdir(target)
    print(request.files.getlist("file"))
    for upload in request.files.getlist("file"):
        print(upload)
        print("{} is the file name".format(upload.filename))
        filename = upload.filename
        # This is to verify files are supported
        ext = os.path.splitext(filename)[1]
        if (ext == ".jpg") or (ext == ".png"):
            print("File supported moving on...")
        else:
            render_template("Error.html", message="Files uploaded are not supported...")


        scale = 4
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

        fa = FaceAligner(predictor, desiredFaceWidth=256)
        #folders = glob.glob('\\images')
        #imagenames__list = []
        #for folder in folders:
        #    for f in glob.glob(folder+'/*.jpg'):
        #        imagenames_list.append(f)

        #read_images = []        

        #for image in imagenames_list:
        #    read_images.append(cv2.imread(image));
        img =  cv2.imread("ag.jpg");
        height, width = img.shape[:2]
        s_height, s_width = height // scale, width // scale
        image = imutils.resize(img, width=800)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cv2.imshow("Input", image)
        rects = detector(gray, 2)
        
        
        for rect in rects:
            (x, y, w, h) = rect_to_bb(rect)
            faceOrigin = imutils.resize(image[y:y + h, x:x + w], width=256)
            faceAligned = fa.align(image, gray, rect)
            cv2.imshow("Original", faceOrigin)
            cv2.imshow("Aligned", faceAligned)
            cv2.waitKey(0)
        destination = "/".join([target, filename])
        print("Accept incoming file:", filename)
        print("Save it to:", destination)
        upload.save(destination)

    # return send_from_directory("images", filename, as_attachment=True)
    return render_template("complete.html", image_name=filename)

#@app.route("/alignment", methods=["POST"])
#def alignment():
#        scale = 4
#        detector = dlib.get_frontal_face_detector()
#        predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

#        fa = aligndlib(predictor, desiredfacewidth=256)
#        img =  cv2.imread("ag.jpg");
#        height, width = img.shape[:2]
#        s_height, s_width = height // scale, width // scale
#        image = imutils.resize(img, width=800)
#        gray = cv2.cvtcolor(image, cv2.color_bgr2gray)
#        cv2.imshow("input", image)
#        rects = detector(gray, 2)
        
#        for rect in rects:
#            (x, y, w, h) = aligndlib.getlargestfaceboundingbox(rect)
#            faceorigin = imutils.resize(image[y:y + h, x:x + w], width=256)
#            facealigned = fa.align(image, gray, rect)
#            cv2.imshow("original", faceorig)
#            cv2.imshow("aligned", facealigned)
#            cv2.waitkey(0)

@app.route('/upload/<filename>')
def send_image(filename):
    return send_from_directory("images", filename)


@app.route('/gallery')
def get_gallery():
    image_names = os.listdir('./images')
    print(image_names)
    return render_template("gallery.html", image_names=image_names)


if __name__ == "__main__":
    app.run(port=4555, debug=True)

