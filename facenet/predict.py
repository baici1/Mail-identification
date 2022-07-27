from PIL import Image
import requests as req
from facenet import Facenet
from io import BytesIO
from flask import Flask, request
from flask import jsonify

app = Flask(__name__)


@app.route("/predict_face", methods=["POST"])
def face_file():
    # 获取普通参数
    print(f"request.values: {request.form}")
    print(f"request.values: {request.values}")
    face1 = request.form.get("face1")
    print(face1)
    response_1 = req.get(face1)
    image_1 = Image.open(BytesIO(response_1.content))
    face2 = request.form.get("face2")
    response_2 = req.get(face2)
    image_2 = Image.open(BytesIO(response_2.content))
    model = Facenet()
    probability = model.detect_image(image_1, image_2)
    print(probability)
    return jsonify({"data": str(probability[-1])})


if __name__ == "__main__":
    app.config["JSON_AS_ASCII"] = False
    app.run(port=5000, debug=True)
    # model = Facenet()

    # while True:
    #     image_1 = input("Input image_1 filename:")
    #     response = req.get(image_1)
    #     try:
    #         image_1 = Image.open(BytesIO(response.content))
    #     except Exception:
    #         print("Image_1 Open Error! Try again!")
    #         continue

    #     image_2 = input("Input image_2 filename:")
    #     try:
    #         image_2 = Image.open(image_2)
    #     except Exception:
    #         print("Image_2 Open Error! Try again!")
    #         continue

    #     probability = model.detect_image(image_1, image_2)
    #     print(probability)
