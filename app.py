# import os
# import logging
# import warnings
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# logging.getLogger('tensorflow').setLevel(logging.ERROR)
# from absl import logging as absl_logging
# absl_logging.set_verbosity(absl_logging.ERROR)
# logging.getLogger('werkzeug').setLevel(logging.ERROR)
# from urllib3.exceptions import NotOpenSSLWarning
# warnings.filterwarnings("ignore", category=NotOpenSSLWarning)


from flask import Flask, request, render_template, redirect, url_for
import tensorflow as tf
import numpy as np
from PIL import Image
import io, os, base64, csv
from datetime import datetime


app = Flask(__name__)


MODEL = tf.keras.models.load_model("model/mobilenetv2_finetuned.h5")
with open("model/class_names.txt") as f:
    CLASS_NAMES = [l.strip() for l in f if l.strip()]

def prepare_image(bytestr, target_size=(224, 224)):
    img = Image.open(io.BytesIO(bytestr)).convert("RGB")
    img = img.resize(target_size)
    arr = np.array(img) / 255.0
    return np.expand_dims(arr, 0)

def b64_image(bytestr, mime="image/jpeg"):
    return f"data:{mime};base64,{base64.b64encode(bytestr).decode()}"


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        up = request.files.get("file")
        if not up:
            return redirect(request.url)

        img_bytes = up.read()
        # make image_data for inline preview
        image_data = b64_image(img_bytes, up.content_type)

        # run prediction
        preds = MODEL.predict(prepare_image(img_bytes))[0]
        idx = int(np.argmax(preds))
        class_name = CLASS_NAMES[idx]
        confidence = float(preds[idx])

        return render_template(
            "index.html",
            image_data=image_data,
            prediction=class_name,
            confidence=f"{confidence:.2%}",
        )

    # GET – check if we were redirected here after feedback
    thank_you = request.args.get("thank_you") == "1"
    return render_template("index.html", thank_you=thank_you)

# —————————————————————————————————————————————————————————
@app.route("/feedback", methods=["POST"])
def feedback():
    # log feedback
    pred = request.form["predicted_class"]
    vote = request.form["feedback"]           # basically "correct" or "incorrect"
    ts   = datetime.utcnow().isoformat(timespec="seconds")

    os.makedirs("logs", exist_ok=True)
    with open("logs/feedback.csv", "a", newline="") as f:
        csv.writer(f).writerow([ts, pred, vote])

    # redirect back to "/" with a flag so we show “Thanks!”
    return redirect(url_for("index", thank_you=1))

# —————————————————————————————————————————————————————————
if __name__ == "__main__":
    app.run(debug=True)
