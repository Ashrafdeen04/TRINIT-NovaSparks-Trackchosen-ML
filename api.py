import io
from flask import Flask, jsonify, request, send_file
from flask_restful import Resource, Api
from sklearn.feature_extraction.text import TfidfVectorizer
from ultralytics import YOLO
import cv2
import os
import numpy as np
import joblib
from ultralytics import YOLO
import threading
import time
import pandas as pd

app = Flask(__name__)
api = Api(app)

class Model(Resource):

    def post(self):
        try:
            print('entered')
            model = YOLO("best.pt")
            class_names = "fault"
            print('model opened')
            if 'file' not in request.files:
                return jsonify({'error': 'No file part'})
            print('jsonified')
            file = request.files['file']

           
            if file.filename == '':
                return jsonify({'error': 'No selected file'})
            print('error not occured')
           
            image_np = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
            img = cv2.resize(image_np, (1020, 500))
            h, w, _ = img.shape
            results = model.predict(img)
            print('predicted')
            for r in results:
                boxes = r.boxes  
                masks = r.masks 
    
            print('boxes detected')
            if masks is not None:
                masks = masks.data.cpu()
                for seg, box in zip(masks.data.cpu().numpy(), boxes):
                    seg = cv2.resize(seg, (w, h))
                    contours, _ = cv2.findContours((seg).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    for contour in contours:
                        d = int(box.cls)
                        c = class_names[d]
                        x, y, x1, y1 = cv2.boundingRect(contour)
                        cv2.polylines(img, [contour], True, color=(0, 0, 255), thickness=2)
                        cv2.putText(img, c, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            print('image shown')
            print('image shown')
            _, img_encoded = cv2.imencode('.png',img)
            print('image shown')
            response = img_encoded.tobytes()
            print('returning')
            return send_file(
                io.BytesIO(response),
                mimetype='image/png',
                as_attachment=True,
                download_name='grayscale_image.png'
            )

        except Exception as e:
            return jsonify({'error': str(e)})
class VideoCapture(Resource):
    def post(Self):
        try:
            model = YOLO("best.pt")
            class_names = model.names
            cap = cv2.VideoCapture(0)

            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))  

            count = 0

            while True:
                ret, img = cap.read()
                if not ret:
                    break
                count += 1
                if count % 3 != 0:
                    continue

                img = cv2.resize(img, (1020, 500))
                h, w, _ = img.shape
                results = model.predict(img)

                for r in results:
                    boxes = r.boxes
                    masks = r.masks

                if masks is not None:
                    masks = masks.data.cpu()
                    for seg, box in zip(masks.data.cpu().numpy(), boxes):
                        seg = cv2.resize(seg, (w, h))
                        contours, _ = cv2.findContours((seg).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                        for contour in contours:
                            d = int(box.cls)
                            c = class_names[d]
                            x, y, x1, y1 = cv2.boundingRect(contour)
                            cv2.polylines(img, [contour], True, color=(0, 0, 255), thickness=2)
                            cv2.putText(img, c, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                out.write(img)

                cv2.imshow('img', img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            cap.release()
            out.release()
            cv2.destroyAllWindows()
        except Exception as e:
            return jsonify({'error': str(e)})
api.add_resource(Model, '/model')
api.add_resource(VideoCapture, '/videocapture')
if __name__ == '__main__':
    app.run(debug=True)
