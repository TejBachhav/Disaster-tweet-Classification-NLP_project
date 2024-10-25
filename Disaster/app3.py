import os
import pandas as pd
import numpy as np
import re
import tensorflow as tf
from flask import Flask, request, jsonify, render_template, send_file
from tensorflow import keras
from tensorflow.keras.utils import custom_object_scope
from custom_layer import PositionalEmbedding, TransformerEncoder  # Import only custom layers
import io
from io import StringIO
from text_vectorization import create_text_vectorization_layer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Load English stopwords
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    stop_words = set(stopwords.words('english'))
except Exception as e:
    stop_words = set()  # Fallback if nltk is not available
    print(f"Error loading NLTK stopwords: {str(e)}")

# Preprocessing function to clean tweet text
def preprocess_tweet(text):
    """Preprocess the tweet text by applying various transformations."""
    text = text.lower()
    return text

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'  # Folder to store uploaded files

# Load the train dataset
train_dataset_path = r"C:\Users\Tej Bachhav\OneDrive\Documents\NLP\Disaster\train_dataset.csv"
train_dataset = pd.read_csv(train_dataset_path)

# Create the text vectorization layer using the train dataset
train_tf, text_vectorization_layer, vocab = create_text_vectorization_layer(train_dataset)

# Build the text vectorization model
text_vectorization_model = keras.Sequential([text_vectorization_layer])
text_vectorization_model.build(input_shape=(None,))
text_vectorization_model.load_weights(r"C:\Users\Tej Bachhav\OneDrive\Documents\NLP\Disaster\text_vectorization_layer.weights.h5")

# Verify the loaded model
text_vectorization_model.summary()

# Load the trained model using the custom layers (PositionalEmbedding, TransformerEncoder)
model_path = os.path.join(r"C:\Users\Tej Bachhav\OneDrive\Documents\NLP\Disaster", "path_to_your_model.h5")
try:
    with custom_object_scope({'PositionalEmbedding': PositionalEmbedding, 'TransformerEncoder': TransformerEncoder}):
        model = keras.models.load_model(model_path)
    print("Model loaded successfully.")
except Exception as e:
    raise ValueError(f"Error loading model: {str(e)}")

# Global variable to store predictions for downloading
global_predictions = []

# Function to classify tweets using the trained model
def classify_tweet(texts, ids=None):
    """Classify a list of tweets as 'Disaster' or 'Not Disaster' based on the trained model."""
    global global_predictions
    try:
        # Vectorize the tweets using the TextVectorization layer
        vectorized_texts = vectorize_texts(texts)

        # Make predictions using the loaded model
        predictions = model.predict(vectorized_texts)

        # Format predictions with labels and probabilities
        results = []
        for i, prediction in enumerate(predictions):
            prediction_value = prediction[0]
            label = "Disaster" if prediction_value >= 0.5 else "Not Disaster"  # Classification threshold
            tweet_id = ids[i] if ids else i + 1  # Use tweet ID if provided
            results.append({
                "id": tweet_id,
                "tweet": texts[i],
                "probability": float(prediction_value),
                "label": label
            })
        global_predictions = results  # Store predictions for later download
        return results

    except Exception as e:
        print(f"Error in classification: {str(e)}")
        return [{"tweet": text, "probability": None, "label": "Error"} for text in texts]

# Vectorization function using TextVectorization layer
def vectorize_texts(texts):
    """Vectorize the texts using the TextVectorization layer."""
    processed_texts = [preprocess_tweet(text) for text in texts]  # Preprocess the texts
    # Instead of using predict, we use the layer directly
    vectorized_texts = text_vectorization_layer(tf.convert_to_tensor(processed_texts))
    return vectorized_texts

# Route to process uploaded test.csv file
@app.route('/process_test', methods=['POST'])
def process_test():
    """Route to process test.csv and prepare it for predictions."""
    try:
        # Check if a file is uploaded
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded."}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected."}), 400

        # Process the uploaded CSV file
        if file and file.filename.endswith('.csv'):
            stream = io.StringIO(file.stream.read().decode("UTF8"), newline=None)
            test_data = pd.read_csv(stream)

            # Ensure "text" column is present in the CSV
            if 'text' not in test_data.columns:
                return jsonify({"error": 'CSV must contain a "text" column.'}), 400
            test_data.fillna("0", inplace=True)

            # Tokenizing text data and removing stopwords
            test_data['tokens'] = test_data['text'].apply(lambda x: word_tokenize(x.lower()))
            test_data['tokens'] = test_data['tokens'].apply(
                lambda x: [word for word in x if word.isalpha() and word not in stop_words]
            )

            # Convert the tokenized text to a TensorFlow Dataset
            test_tf = tf.data.Dataset.from_tensor_slices(test_data['tokens'].tolist())
            
            # Batch the test data
            BATCH = 32  # Define your batch size
            test_tf = test_tf.batch(BATCH)
            
            # Vectorize the test dataset
            test_tf = test_tf.map(lambda twt: text_vectorization_layer(twt), num_parallel_calls=tf.data.AUTOTUNE)

            # Make predictions using the loaded model
            predictions = model.predict(test_tf)

            # Format predictions with labels and probabilities
            results = []
            for i, prediction in enumerate(predictions):
                prediction_value = prediction[0]
                label = "Disaster" if prediction_value >= 0.5 else "Not Disaster"  # Classification threshold
                results.append({
                    "tweet": test_data['text'].iloc[i],  # Original tweet text
                    "probability": float(prediction_value),
                    "label": label
                })
            return jsonify({"results": results}), 200

        return jsonify({"error": "Uploaded file is not a CSV."}), 400

    except Exception as e:
        print(f"Error processing test file: {str(e)}")
        return jsonify({"error": str(e)}), 500

# Main route to render the homepage
@app.route('/')
def home():
    return render_template('index.html')

# Route to handle single or batch predictions
@app.route('/predict', methods=['POST'])
def predict_route():
    """Route to handle single or batch tweet predictions via textarea or CSV."""
    global global_predictions
    try:
        if 'file' in request.files:
            file = request.files['file']
            if file.filename == '':
                return render_template('index.html', error='No file selected.')

            if file and file.filename.endswith('.csv'):
                stream = io.StringIO(file.stream.read().decode("UTF8"), newline=None)
                csv_input = pd.read_csv(stream)

                # Ensure the CSV has 'text' and 'id' columns
                if 'text' in csv_input.columns and 'id' in csv_input.columns:
                    tweet_list = csv_input['text'].tolist()
                    ids_list = csv_input['id'].tolist()  # Extract IDs from CSV
                    result = classify_tweet(tweet_list, ids=ids_list)
                    return render_template('result.html', predictions=result)
                else:
                    return render_template('index.html', error='CSV file must contain "text" and "id" columns.')

        # Handle input from the textarea
        tweets = request.form.get('tweets', '')
        if not tweets:
            return render_template('index.html', error='No tweet provided.')

        tweet_list = [tweet.strip() for tweet in tweets.splitlines() if tweet.strip()]  # Process textarea input
        result = classify_tweet(tweet_list)

        return render_template('result.html', predictions=result)

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        return render_template('index.html', error=str(e))

# Route to download the full predictions as CSV
@app.route('/download_full', methods=['GET'])
def download_full():
    """Download the full results as a CSV file."""
    global global_predictions
    if not global_predictions:
        return "No results to download.", 400

    df = pd.DataFrame(global_predictions)
    
    output = StringIO()
    df.to_csv(output, index=False)
    output.seek(0)

    return send_file(io.BytesIO(output.getvalue().encode()), download_name='full_results.csv', as_attachment=True)

# Route to download only the ID and target as CSV
@app.route('/download_targets', methods=['GET'])
def download_targets():
    """Download only the ID and target as a CSV file."""
    global global_predictions
    if not global_predictions:
        return "No results to download.", 400

    ids_and_targets = []
    for result in global_predictions:
        target = 1 if result['label'] == "Disaster" else 0
        ids_and_targets.append({"id": result['id'], "target": target})

    df = pd.DataFrame(ids_and_targets)
    
    output = StringIO()
    df.to_csv(output, index=False)
    output.seek(0)

    return send_file(io.BytesIO(output.getvalue().encode()), download_name='targets_results.csv', as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
