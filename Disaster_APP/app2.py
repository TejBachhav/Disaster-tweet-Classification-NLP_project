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
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)  # Remove URLs
    text = re.sub(r'\@\w+|\#', '', text)  # Remove mentions, hashtags
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    tokens = word_tokenize(text)  # Tokenize the text
    tokens = [word for word in tokens if word.isalpha() and word not in stop_words]  # Remove stopwords
    return ' '.join(tokens)

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'  # Folder to store uploaded files

# Load the train dataset
train_dataset_path = r"C:\Users\Tej Bachhav\OneDrive\Documents\NLP\Disaster\train_dataset.csv"
train_dataset = pd.read_csv(train_dataset_path)

# Create the text vectorization layer using the train dataset
train_tf, text_vectorization_layer, vocab = create_text_vectorization_layer(train_dataset)

# Build and load TextVectorization layer's weights
text_vectorization_model = keras.Sequential([text_vectorization_layer])
text_vectorization_model.build(input_shape=(None,))
text_vectorization_model.load_weights(r"C:\Users\Tej Bachhav\OneDrive\Documents\NLP\Disaster\text_vectorization_layer.weights.h5")

# Load the trained model using the custom layers (PositionalEmbedding, TransformerEncoder)
model_path = os.path.join(r"C:\Users\Tej Bachhav\OneDrive\Documents\NLP\Disaster", "path_to_your_model.h5")
with custom_object_scope({'PositionalEmbedding': PositionalEmbedding, 'TransformerEncoder': TransformerEncoder}):
    model = keras.models.load_model(model_path)

# Global variable to store predictions for downloading
global_predictions = []

# Function to classify tweets using the trained model
def classify_tweet(texts, ids=None):
    """Classify a list of tweets as 'Disaster' or 'Not Disaster' based on the trained model."""
    global global_predictions
    try:
        vectorized_texts = vectorize_texts(texts)  # Vectorize tweets
        predictions = model.predict(vectorized_texts)  # Get predictions

        results = []
        for i, prediction in enumerate(predictions):
            label = "Disaster" if prediction[0] >= 0.5 else "Not Disaster"
            tweet_id = ids[i] if ids else i + 1
            results.append({
                "id": tweet_id,
                "tweet": texts[i],
                "probability": float(prediction[0]),
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
    processed_texts = [preprocess_tweet(text) for text in texts]
    return text_vectorization_layer(tf.convert_to_tensor(processed_texts))

# Route for single tweet classification
@app.route('/classify_single', methods=['POST'])
def classify_single():
    """Route to classify a single tweet."""
    global global_predictions
    tweet = request.form.get('tweet', '')
    if not tweet:
        return jsonify({"error": "No tweet provided."}), 400
    result = classify_tweet([tweet])  # Classify the single tweet
    return render_template('result.html', predictions=result)

# Route for multiple tweet classification
@app.route('/classify_multiple', methods=['POST'])
def classify_multiple():
    """Classify multiple tweets."""
    global global_predictions
    try:
        tweets = request.form.get('tweets', '')
        if not tweets:
            return jsonify({"error": "No tweets provided."}), 400
        tweet_list = [tweet.strip() for tweet in tweets.splitlines() if tweet.strip()]
        results = classify_tweet(tweet_list)
        return render_template('result.html', predictions=results)
    except Exception as e:
        print(f"Error in multiple tweet classification: {str(e)}")
        return jsonify({"error": str(e)}), 500

# Route to process uploaded test.csv file
@app.route('/process_test', methods=['POST'])
def process_test():
    """Route to process test.csv and prepare it for predictions."""
    global global_predictions
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded."}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected."}), 400
        if file and file.filename.endswith('.csv'):
            stream = io.StringIO(file.stream.read().decode("UTF8"), newline=None)
            test_data = pd.read_csv(stream)

            if 'text' not in test_data.columns:
                return jsonify({"error": 'CSV must contain a "text" column.'}), 400

            # Check if 'id' column exists; if not, create one
            if 'id' not in test_data.columns:
                test_data['id'] = range(1, len(test_data) + 1)  # Generate IDs if not present

            test_data.fillna("0", inplace=True)
            test_data['processed_text'] = test_data['text'].apply(preprocess_tweet)

            # Convert to TensorFlow Dataset and batch
            BATCH = 32
            test_tf = tf.data.Dataset.from_tensor_slices(test_data['processed_text'].tolist()).batch(BATCH)
            test_tf = test_tf.map(lambda twt: text_vectorization_layer(twt), num_parallel_calls=tf.data.AUTOTUNE)

            predictions = model.predict(test_tf)

            results = []
            for i, prediction in enumerate(predictions):
                label = "Disaster" if prediction[0] >= 0.5 else "Not Disaster"
                results.append({
                    "id": test_data['id'].iloc[i],  # Store the corresponding ID
                    "tweet": test_data['text'].iloc[i],
                    "probability": float(prediction[0]),
                    "label": label
                })
            
            global_predictions = results
            return render_template('result.html', predictions=results)
        return jsonify({"error": "Uploaded file is not a CSV."}), 400
    except Exception as e:
        print(f"Error processing test file: {str(e)}")
        return jsonify({"error": str(e)}), 500


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

    df_targets = pd.DataFrame(ids_and_targets)
    
    output = StringIO()
    df_targets.to_csv(output, index=False)
    output.seek(0)

    return send_file(io.BytesIO(output.getvalue().encode()), download_name='targets.csv', as_attachment=True)


# Main route to render the homepage
@app.route('/')
def home():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
