from flask import Flask, request, jsonify, render_template, send_from_directory
import os
from werkzeug.utils import secure_filename
from pyngrok import ngrok
from tensorflow.keras.layers import Input, Dense, Dropout, concatenate, GlobalAveragePooling1D, Lambda
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from transformers import BertTokenizer, TFBertModel
import pytesseract
from PIL import Image
import numpy as np
import tensorflow as tf
import re

# Set up the upload folder path
UPLOAD_FOLDER = 'uploads'
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


# Transformer block for ViT
def transformer_block(inputs, num_heads, ff_dim, dropout_rate):
    attention_output = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=inputs.shape[-1])(inputs, inputs)
    attention_output = tf.keras.layers.Dropout(dropout_rate)(attention_output)
    out1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(attention_output + inputs)
    ff_output = Dense(ff_dim, activation='relu')(out1)
    ff_output = Dense(inputs.shape[-1])(ff_output)
    ff_output = tf.keras.layers.Dropout(dropout_rate)(ff_output)
    return tf.keras.layers.LayerNormalization(epsilon=1e-6)(ff_output + out1)

# Rebuild ViT model
def create_vit_model(input_shape=(224, 224, 3), num_heads=8, ff_dim=128, num_transformer_blocks=4, num_classes=1):
    inputs = Input(shape=input_shape)
    patches = tf.keras.layers.Conv2D(filters=64, kernel_size=(16, 16), strides=(16, 16), padding='valid')(inputs)
    patches = tf.keras.layers.Reshape((-1, patches.shape[-1]))(patches)
    position_embedding = tf.keras.layers.Embedding(input_dim=patches.shape[1], output_dim=patches.shape[-1])(tf.range(start=0, limit=patches.shape[1], delta=1))
    patches_with_position = patches + position_embedding
    for _ in range(num_transformer_blocks):
        patches_with_position = transformer_block(patches_with_position, num_heads, ff_dim, dropout_rate=0.1)
    representation = GlobalAveragePooling1D()(patches_with_position)
    x = Dense(256, activation='relu')(representation)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation='sigmoid')(x)
    return Model(inputs=inputs, outputs=outputs)

# BERT-based text classification model
def create_bert_model():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = TFBertModel.from_pretrained('bert-base-uncased')
    
    input_ids = Input(shape=(128,), dtype=tf.int32, name="input_ids")
    attention_mask = Input(shape=(128,), dtype=tf.int32, name="attention_mask")
    
    def bert_layer(inputs):
        return bert_model(input_ids=inputs[0], attention_mask=inputs[1]).last_hidden_state
    
    bert_output = Lambda(bert_layer, output_shape=(128, 768))([input_ids, attention_mask])
    cls_token_embedding = bert_output[:, 0, :]
    
    dense = Dense(64, activation='relu')(cls_token_embedding)
    dense = Dropout(0.5)(dense)
    output = Dense(1, activation='sigmoid')(dense)
    
    return Model(inputs=[input_ids, attention_mask], outputs=output)

# BERT-based embeddings
def get_real_bert_embeddings(input_ids, attention_mask):
    bert_model = TFBertModel.from_pretrained('bert-base-uncased')  # Load the pre-trained BERT model
    outputs = bert_model(input_ids=input_ids, attention_mask=attention_mask)
    
    # Use CLS token for embeddings 
    cls_token_embedding = outputs.last_hidden_state[:, 0, :] 
    return cls_token_embedding


#  Multimodal model
def create_multimodal_model():
    # Image input for ViT
    vit_model = create_vit_model()  
    image_input = vit_model.input  

    # Precomputed BERT embeddings input
    bert_embeddings_input = Input(shape=(768,), name="bert_embeddings_input")  

    # Get outputs from ViT and BERT
    vit_output = vit_model.output  # Output from ViT model
    bert_output = bert_embeddings_input  # Precomputed BERT embeddings

    # Concatenate ViT and BERT outputs
    combined = concatenate([vit_output, bert_output])

    # Fully connected layers after concatenation
    combined_dense = Dense(128, activation='relu')(combined)
    combined_dense = Dropout(0.5)(combined_dense)
    output = Dense(1, activation='sigmoid')(combined_dense)

    # Final multimodal model with two inputs
    multimodal_model = Model(inputs=[image_input, bert_embeddings_input], outputs=output)
    return multimodal_model

# Create the multimodal model
model = create_multimodal_model()
model.compile(optimizer=Adam(learning_rate=1e-5), loss='binary_crossentropy', metrics=['accuracy'])

# Load BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Preprocess text using BERT tokenizer
def preprocess_text(text):
    inputs = tokenizer(text, return_tensors='tf', truncation=True, padding=True, max_length=128)
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    return input_ids, attention_mask

# Preprocess uploaded image for ViT
def preprocess_image(image):
    image = image.resize((224, 224))  # Resize image to the expected input size
    image = np.array(image) / 255.0   # Normalize the image
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# OCR to extract text from the uploaded image
def extract_text_from_image(image):
    import pytesseract
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    return pytesseract.image_to_string(image)


# Route to render the HTML upload page
@app.route("/", methods=["GET"])
def home():
    return render_template("results.html")

@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route("/predict", methods=["POST"])
def predict():
    # Check if an image file is included in the request
    if 'file' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    # Get the uploaded image file
    file = request.files['file']
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)
    
    image = Image.open(file_path)

    # Preprocess the image for ViT
    processed_image = preprocess_image(image)

    # Extract text from the image using OCR
    extracted_text = extract_text_from_image(image)

    # Preprocess the extracted text using BERT tokenizer
    input_ids, attention_mask = preprocess_text(extracted_text)

    # Use real BERT embeddings instead of random ones
    bert_embeddings = get_real_bert_embeddings(input_ids, attention_mask)

    print("Running prediction")
    # Run the preprocessed image and BERT embeddings through the model
    prediction = model.predict([processed_image, bert_embeddings])
    print("Prediction complete")

    # Interpret the result, binary classification
    result = 'Hate Speech Detected' if prediction[0] > 0.5 else 'No Hate Speech Detected'

    # Return the result as JSON, including the filename for the image display
    return jsonify({'result': result, 'extracted_text': extracted_text, 'filename': filename})



if __name__ == "__main__":
    print("Starting the Flask app...")
    app.run(debug=True)
