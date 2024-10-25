import tensorflow as tf
from tensorflow.keras.layers import TextVectorization
import numpy as np

SEED = 4243

class CustomTextVectorization(TextVectorization):
    def __init__(self, max_tokens=20000, output_mode='int', output_sequence_length=165, **kwargs):
        super().__init__(max_tokens=max_tokens, output_mode=output_mode, output_sequence_length=output_sequence_length, **kwargs)

    def adapt(self, data):
        super().adapt(data)

    def call(self, inputs):
        return super().call(inputs)

# Function to create the text vectorization layer
def create_text_vectorization_layer(train_dataset, target_column='target', batch_size=32):
    # Concatenate the text data of the columns "keyword", "location", "text"
    train_tf = tf.data.Dataset.from_tensor_slices(
        (train_dataset['keyword'] + train_dataset['location'] + train_dataset['text'], train_dataset[target_column])
    )

    # Convert the data into batches
    train_tf = train_tf.shuffle(int((SEED * 13) / 8)).batch(batch_size)

    # Instantiate the TextVectorization layer
    max_length = 165
    max_tokens = 20_000
    vectorization_layer = CustomTextVectorization(max_tokens=max_tokens,
                                                  output_mode='int',
                                                  output_sequence_length=max_length)

    # Learn the vocabulary
    vectorization_layer.adapt(train_tf.map(lambda twt, target: twt))

    # Get the vocabulary
    vocab = vectorization_layer.get_vocabulary()
    print("Vocabulary size =", len(vocab))

    # Convert the list object to NumPy array for decoding the vectorized data
    vocab = np.array(vocab)

    # Vectorize the train dataset
    train_tf = train_tf.map(lambda twt, target: (vectorization_layer(twt), target),
                            num_parallel_calls=tf.data.AUTOTUNE)

    return train_tf, vectorization_layer, vocab

# Function to create a test dataset
def create_test_dataset(test, text_vectorization, batch_size=32):
    """Creates a test dataset from the input DataFrame.

    Args:
        test (DataFrame): The DataFrame containing the test data.
        text_vectorization (CustomTextVectorization): The fitted text vectorization layer.
        batch_size (int): The size of batches.

    Returns:
        tf.data.Dataset: The batched and vectorized test dataset.
    """
    # Concatenate the text data of the columns "keyword", "location", "text"
    test_tf = tf.data.Dataset.from_tensor_slices(
        test['keyword'] + test['location'] + test['text']
    )

    # Convert the data into batches
    test_tf = test_tf.batch(batch_size)

    # Vectorize the test dataset
    test_tf = test_tf.map(lambda twt: text_vectorization(twt),
                          num_parallel_calls=tf.data.AUTOTUNE)

    return test_tf

# Example of how to use these functions
# Assume train_dataset is a DataFrame with the appropriate columns
# train_tf, text_vectorization, vocab = create_text_vectorization_layer(train_dataset)

# Assume test_dataset is a DataFrame with the same columns
# test_tf = create_test_dataset(test_dataset, text_vectorization)
