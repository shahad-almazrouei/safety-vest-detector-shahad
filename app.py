import os
import uuid
from flask import Flask, render_template, request, url_for
from ultralytics import YOLO
from PIL import Image

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
RESULT_FOLDER = "static/results"
MODEL_PATH = "best.pt"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

model = YOLO(MODEL_PATH)

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/", methods=["GET", "POST"])
def index():
    result_image = None
    uploaded_image = None
    prediction_text = None
    error = None

    if request.method == "POST":
        if "image" not in request.files:
            error = "No file part found."
            return render_template("index.html", error=error)

        file = request.files["image"]

        if file.filename == "":
            error = "Please choose an image."
            return render_template("index.html", error=error)

        if file and allowed_file(file.filename):
            ext = file.filename.rsplit(".", 1)[1].lower()
            unique_name = f"{uuid.uuid4().hex}.{ext}"

            upload_path = os.path.join(UPLOAD_FOLDER, unique_name)
            result_path = os.path.join(RESULT_FOLDER, unique_name)

            file.save(upload_path)

            results = model.predict(source=upload_path, conf=0.25, save=False)
            result = results[0]

            plotted = result.plot()
            Image.fromarray(plotted[..., ::-1]).save(result_path)

            class_names = model.names
            counts = {}

            if result.boxes is not None and len(result.boxes) > 0:
                for cls_id in result.boxes.cls.tolist():
                    cls_name = class_names[int(cls_id)]
                    counts[cls_name] = counts.get(cls_name, 0) + 1

                prediction_text = ", ".join(
                    [f"{label}: {count}" for label, count in counts.items()]
                )
            else:
                prediction_text = "No objects detected."

            uploaded_image = url_for("static", filename=f"uploads/{unique_name}")
            result_image = url_for("static", filename=f"results/{unique_name}")

            return render_template(
                "index.html",
                uploaded_image=uploaded_image,
                result_image=result_image,
                prediction_text=prediction_text
            )

        error = "Invalid file type. Please upload PNG, JPG, or JPEG."

    return render_template("index.html", error=error)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)