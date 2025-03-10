from flask import Flask, render_template, request
import os
from steganography import embed_text_lsb, extract_text_lsb

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
OUTPUT_FOLDER = "static/output"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        action = request.form["action"]
        uploaded_file = request.files["image"]
        if uploaded_file:
            image_path = os.path.join(UPLOAD_FOLDER, uploaded_file.filename)
            uploaded_file.save(image_path)

            if action == "embed":
                text = request.form["text"]
                stego_image_path, grid_image_path = embed_text_lsb(image_path, text, OUTPUT_FOLDER)
                return render_template("index.html", stego_image=stego_image_path, grid_image=grid_image_path)

            elif action == "extract":
                extracted_text, grid_image_path = extract_text_lsb(image_path, OUTPUT_FOLDER)
                return render_template("index.html", extracted_text=extracted_text, grid_image=grid_image_path)

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)