import os
import cv2
import numpy as np
import math
from flask import Flask, render_template, request, url_for, send_file
from werkzeug.utils import secure_filename

app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
STEG_FOLDER = "static/stego"
VIS_FOLDER = "static/vis"  # For visualization images
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["STEG_FOLDER"] = STEG_FOLDER
app.config["VIS_FOLDER"] = VIS_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STEG_FOLDER, exist_ok=True)
os.makedirs(VIS_FOLDER, exist_ok=True)

##########################################
# GRID & ENTROPY FUNCTIONS
##########################################

def choose_nblocks(image):
    """Choose a total number of blocks based on the image's smallest dimension."""
    h, w = image.shape[:2]
    min_dim = min(h, w)
    if min_dim < 256:
        return 8
    elif min_dim < 512:
        return 16
    elif min_dim < 1024:
        return 32
    elif min_dim < 2048:
        return 64
    else:
        return 128

def get_grid_dimensions(nblocks):
    """Compute grid dimensions (rows, cols) given a desired total number of blocks."""
    rows = int(math.sqrt(nblocks))
    cols = int(math.ceil(nblocks / rows))
    return rows, cols

def divide_image_grid(image, nblocks):
    """
    Divide image into blocks based on grid dimensions computed from nblocks.
    Returns a list of tuples: (block, (x, y, width, height)) and the grid dims.
    """
    h, w = image.shape[:2]
    rows, cols = get_grid_dimensions(nblocks)
    block_h = h // rows
    block_w = w // cols
    blocks = []
    for i in range(rows):
        for j in range(cols):
            x1 = j * block_w
            y1 = i * block_h
            x2 = x1 + block_w if j < cols - 1 else w
            y2 = y1 + block_h if i < rows - 1 else h
            block = image[y1:y2, x1:x2]
            blocks.append((block, (x1, y1, x2 - x1, y2 - y1)))
    return blocks, (rows, cols)

def shannon_entropy(block):
    """Compute Shannon entropy for a given grayscale block."""
    hist = cv2.calcHist([block], [0], None, [256], [0, 256])
    total = hist.sum()
    if total == 0:
        return 0
    hist = hist.ravel() / total
    hist = hist[hist > 0]
    return -np.sum(hist * np.log2(hist))

def find_top_entropy_blocks(image, nblocks, top_n=2):
    """
    Divide the grayscale image into blocks (using grid defined by nblocks),
    compute entropy for each block, and return the coordinates (x, y, w, h)
    of the top N blocks along with the grid dimensions.
    """
    blocks, grid_dims = divide_image_grid(image, nblocks)
    entropy_values = []
    for block, coords in blocks:
        if block.size > 0:
            ent = shannon_entropy(block)
            entropy_values.append((ent, coords))
    if len(entropy_values) < top_n:
        return [], grid_dims
    top_blocks = sorted(entropy_values, key=lambda x: x[0], reverse=True)[:top_n]
    return [coords for (ent, coords) in top_blocks], grid_dims

##########################################
# LSB EMBED & EXTRACT FUNCTIONS
##########################################

def embed_text_lsb(image, text, coords):
    """
    Embed text into the LSB of the blue channel of the specified block.
    (A 16-bit EOF marker is appended.)
    """
    x, y, bw, bh = coords
    binary_text = ''.join(format(ord(c), '08b') for c in text) + '1111111111111110'
    index = 0
    for i in range(y, y + bh):
        for j in range(x, x + bw):
            if index < len(binary_text):
                # Modify only the blue channel (channel index 0)
                image[i, j, 0] = (image[i, j, 0] & 0xFE) | int(binary_text[index])
                index += 1
            else:
                return image
    return image

def extract_text_lsb(image, coords):
    """
    Extract text from the LSB of the blue channel of the specified block.
    Extraction stops when a 16-bit EOF marker is found; if not found, use all bits.
    """
    x, y, bw, bh = coords
    bits = ""
    for i in range(y, y + bh):
        for j in range(x, x + bw):
            bits += str(image[i, j, 0] & 1)
    marker = '1111111111111110'
    pos = bits.find(marker)
    if pos == -1:
        pos = len(bits)
    bits = bits[:pos]
    text = ""
    for i in range(0, len(bits), 8):
        byte = bits[i:i+8]
        if len(byte) < 8:
            break
        text += chr(int(byte, 2))
    return text

##########################################
# VISUALIZATION FUNCTION
##########################################

def overlay_grid(image, grid_dims, key_coords, msg_coords):
    """
    Draw the grid on a copy of the image and highlight the key block (red)
    and message block (blue).
    """
    rows, cols = grid_dims
    h, w = image.shape[:2]
    block_h = h // rows
    block_w = w // cols
    vis = image.copy()
    # Draw grid lines
    for i in range(1, rows):
        y = i * block_h
        cv2.line(vis, (0, y), (w, y), (0, 255, 0), 1)
    for j in range(1, cols):
        x = j * block_w
        cv2.line(vis, (x, 0), (x, h), (0, 255, 0), 1)
    # Draw rectangles for key (red) and message (blue)
    kx, ky, kw, kh = key_coords
    cv2.rectangle(vis, (kx, ky), (kx+kw, ky+kh), (0, 0, 255), 2)
    mx, my, mw, mh = msg_coords
    cv2.rectangle(vis, (mx, my), (mx+mw, my+mh), (255, 0, 0), 2)
    return vis

##########################################
# FLASK ROUTES
##########################################

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        action = request.form.get("action")
        # EMBED MODE
        if action == "embed":
            if "image" not in request.files or not request.form.get("key") or not request.form.get("message"):
                return "Error: missing image, key or message."
            file = request.files["image"]
            key = request.form.get("key")
            message = request.form.get("message")
            if file.filename == "":
                return "Error: no file selected."
            filename = secure_filename(file.filename)
            img_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(img_path)
            image = cv2.imread(img_path)
            if image is None:
                return "Error: could not read image."
            nblocks = choose_nblocks(image)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            top_blocks, grid_dims = find_top_entropy_blocks(gray, nblocks, top_n=2)
            if len(top_blocks) < 2:
                return "Error: not enough high-entropy blocks found."
            key_coords, msg_coords = top_blocks[0], top_blocks[1]
            stego_image = image.copy()
            stego_image = embed_text_lsb(stego_image, key, key_coords)
            stego_image = embed_text_lsb(stego_image, message, msg_coords)
            stego_filename = "stego_" + filename
            stego_path = os.path.join(app.config["STEG_FOLDER"], stego_filename)
            cv2.imwrite(stego_path, stego_image)
            vis_image = overlay_grid(image, grid_dims, key_coords, msg_coords)
            vis_filename = "vis_" + filename
            vis_path = os.path.join(app.config["VIS_FOLDER"], vis_filename)
            cv2.imwrite(vis_path, vis_image)
            orig_url = url_for('static', filename="uploads/" + filename)
            stego_url = url_for('static', filename="stego/" + stego_filename)
            vis_url = url_for('static', filename="vis/" + vis_filename)
            return render_template("index.html",
                                   action=action,
                                   orig_url=orig_url,
                                   stego_url=stego_url,
                                   vis_url=vis_url,
                                   extracted_key=None,
                                   extracted_message=None)
        # EXTRACT MODE
        elif action == "extract":
            if "image" not in request.files:
                return "Error: missing image."
            file = request.files["image"]
            if file.filename == "":
                return "Error: no file selected."
            filename = secure_filename(file.filename)
            img_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(img_path)
            image = cv2.imread(img_path)
            if image is None:
                return "Error: could not read image."
            nblocks = choose_nblocks(image)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            top_blocks, grid_dims = find_top_entropy_blocks(gray, nblocks, top_n=2)
            if len(top_blocks) < 2:
                return "Error: not enough high-entropy blocks found."
            key_coords, msg_coords = top_blocks[0], top_blocks[1]
            # Extract text from the stego image (using the same block coordinates)
            extracted_key = extract_text_lsb(image, key_coords)
            extracted_message = extract_text_lsb(image, msg_coords)
            # Debug prints (you can remove these later)
            print("Extracted Key:", extracted_key)
            print("Extracted Message:", extracted_message)
            vis_image = overlay_grid(image, grid_dims, key_coords, msg_coords)
            vis_filename = "vis_" + filename
            vis_path = os.path.join(app.config["VIS_FOLDER"], vis_filename)
            cv2.imwrite(vis_path, vis_image)
            vis_url = url_for('static', filename="vis/" + vis_filename)
            return render_template("index.html",
                                   action=action,
                                   orig_url=None,
                                   stego_url=None,
                                   vis_url=vis_url,
                                   extracted_key=extracted_key,
                                   extracted_message=extracted_message)
    return render_template("index.html",
                           orig_url=None,
                           stego_url=None,
                           vis_url=None,
                           extracted_key=None,
                           extracted_message=None)

@app.route("/download/<folder>/<filename>")
def download_file(folder, filename):
    if folder == "stego":
        path = os.path.join(app.config["STEG_FOLDER"], filename)
    elif folder == "vis":
        path = os.path.join(app.config["VIS_FOLDER"], filename)
    else:
        return "Invalid folder."
    return send_file(path, as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)
