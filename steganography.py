import cv2
import numpy as np
import os
from math import log2

def compute_entropy(block):
    """ Compute Shannon Entropy of an image block. """
    hist = cv2.calcHist([block], [0], None, [256], [0, 256])
    hist = hist / hist.sum()  # Normalize
    entropy = -sum(p * log2(p) for p in hist if p > 0)
    return entropy

def segment_image(image):
    """ Segment image based on resolution & return blocks. """
    h, w = image.shape[:2]
    
    if w >= 2561 and h >= 1921:
        rows, cols = (4, 32) if w > h else (32, 4)
    elif w >= 1281 and h >= 961:
        rows, cols = (4, 16) if w > h else (16, 4)
    elif w >= 641 and h >= 480:
        rows, cols = (4, 8) if w > h else (8, 4)
    else:
        rows, cols = (2, 8) if w > h else (8, 2)

    block_h, block_w = h // rows, w // cols
    blocks = []
    for i in range(rows):
        for j in range(cols):
            block = image[i * block_h:(i + 1) * block_h, j * block_w:(j + 1) * block_w]
            blocks.append(((i, j), block))

    return blocks, (rows, cols)

def draw_grid(image, grid_size, save_path, busiest_block):
    """ Draw a grid on the image and highlight the busiest block. """
    h, w = image.shape[:2]
    rows, cols = grid_size
    block_h, block_w = h // rows, w // cols

    overlay = image.copy()
    for i in range(rows):
        for j in range(cols):
            cv2.rectangle(overlay, (j * block_w, i * block_h), ((j + 1) * block_w, (i + 1) * block_h), (255, 255, 255), 1)
            
    i, j = busiest_block
    cv2.rectangle(overlay, (j * block_w, i * block_h), ((j + 1) * block_w, (i + 1) * block_h), (0, 0, 255), 3)

    cv2.imwrite(save_path, overlay)
    return save_path

def text_to_binary(text):
    """ Convert text to binary. """
    return ''.join(format(ord(c), '08b') for c in text)

def binary_to_text(binary_string):
    """ Convert binary string back to text. """
    chars = [binary_string[i:i+8] for i in range(0, len(binary_string), 8)]
    return ''.join(chr(int(b, 2)) for b in chars if int(b, 2) != 0)

def embed_text_lsb(image_path, text, save_folder):
    """ Hide text in the busiest block using LSB steganography. """
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blocks, grid_size = segment_image(gray)
    busiest_block = max(blocks, key=lambda b: compute_entropy(b[1]))[0]

    i, j = busiest_block
    h, w = image.shape[:2]
    block_h, block_w = h // grid_size[0], w // grid_size[1]

    block = image[i * block_h:(i + 1) * block_h, j * block_w:(j + 1) * block_w]
    
    binary_text = text_to_binary(text) + '00000000'  
    binary_index = 0

    for row in range(block.shape[0]):
        for col in range(block.shape[1]):
            if binary_index < len(binary_text):
                pixel = list(block[row, col])
                pixel[0] = (pixel[0] & ~1) | int(binary_text[binary_index]) 
                block[row, col] = tuple(pixel)
                binary_index += 1

    image[i * block_h:(i + 1) * block_h, j * block_w:(j + 1) * block_w] = block

    stego_path = os.path.join(save_folder, "stego_image.png")
    cv2.imwrite(stego_path, image)

    grid_path = os.path.join(save_folder, "grid_image.png")
    draw_grid(image, grid_size, grid_path, busiest_block)

    return stego_path, grid_path

def extract_text_lsb(image_path, save_folder):
    """ Extract hidden text from the busiest block. """
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blocks, grid_size = segment_image(gray)
    busiest_block = max(blocks, key=lambda b: compute_entropy(b[1]))[0]

    i, j = busiest_block
    h, w = image.shape[:2]
    block_h, block_w = h // grid_size[0], w // grid_size[1]

    block = image[i * block_h:(i + 1) * block_h, j * block_w:(j + 1) * block_w]
    
    binary_text = ''
    for row in range(block.shape[0]):
        for col in range(block.shape[1]):
            binary_text += str(block[row, col, 0] & 1) 
            
            if binary_text[-8:] == "00000000":  
                break
        else:
            continue
        break

    extracted_text = binary_to_text(binary_text[:-8])  

    grid_path = os.path.join(save_folder, "grid_image.png")
    draw_grid(image, grid_size, grid_path, busiest_block)

    return extracted_text, grid_path
