from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras.utils import load_img, img_to_array
import numpy as np
import os

app = Flask(__name__)

model = tf.keras.models.load_model("../plant_disease_model.keras")

class_names = [
    "Healthy",
    "Early Blight",
    "Late Blight"
]

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None

    if request.method == "POST":
        file = request.files["image"]

        if file:
            os.makedirs("uploads", exist_ok=True)

            image_path = os.path.join("uploads", file.filename)
            file.save(image_path)

            image = load_img(image_path, target_size=(128, 128))
            image = img_to_array(image)
            image = image / 255.0
            image = np.expand_dims(image, axis=0)

            result = model.predict(image)
            prediction = class_names[np.argmax(result)]

        return render_template("index.html", prediction=prediction)

    return render_template("index.html", prediction=prediction)


if __name__ == "__main__":
    app.run(debug=True)