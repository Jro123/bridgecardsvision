import platform
from flask import Flask, request, jsonify
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch
import os
import base64
import cv2
import numpy as np

app = Flask(__name__)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

# Détecter le système d'exploitation
# Charger le modèle et le processeur
processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-printed')
model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-printed').to(device)


@app.route('/predict', methods=['POST'])
def predict():
    image_data = request.data
    np_data = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(np_data, cv2.IMREAD_COLOR)
    if image is None:
        return jsonify({'recognized_text': '', 'bbox': [], 'orientation': 0})
    bbox = []

    image384 = cv2.resize(image, (384, 384))
    # Préparer l'image pour le modèle
    pixel_values = processor(images=image384, return_tensors="pt").pixel_values.to(device)
    outputs = model.generate(
        pixel_values,
        max_length=2,
        return_dict_in_generate=True,
        output_scores=True
    )
    generated_ids = outputs.sequences
    #generated_ids = model.generate(pixel_values)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    logits = outputs.scores 
    confidence_scores = [torch.softmax(logit, dim=-1) for logit in logits] 
    avg_confidence_score = np.mean([score.max().item() for score in confidence_scores])
    
    print('texte : [',generated_text, ']')

    result = {
        'recognized_text': generated_text,
        'confidence': float(avg_confidence_score),
        'bbox': bbox,
        'orientation': 0
    }
    print(result)
    return jsonify(result)
    #return jsonify({'recognized_text': generated_text})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)
