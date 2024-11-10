# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import layers
# import numpy as np
# import pandas as pd

# # Define custom PositionalEmbedding layer
# class PositionalEmbedding(layers.Layer):
#     def __init__(self, sequence_length, input_dim, output_dim, **kwargs):
#         super().__init__(**kwargs)
#         self.token_embeddings = layers.Embedding(input_dim=input_dim, output_dim=output_dim)
#         self.position_embeddings = layers.Embedding(input_dim=sequence_length, output_dim=output_dim)
#         self.sequence_length = sequence_length
#         self.input_dim = input_dim
#         self.output_dim = output_dim

#     def call(self, inputs):
#         length = tf.shape(inputs)[-1]
#         positions = tf.range(start=0, limit=length, delta=1)
#         embedded_tokens = self.token_embeddings(inputs)
#         embedded_positions = self.position_embeddings(positions)
#         return embedded_tokens + embedded_positions

#     def get_config(self):
#         config = super().get_config()
#         config.update({
#             "sequence_length": self.sequence_length,
#             "input_dim": self.input_dim,
#             "output_dim": self.output_dim,
#         })
#         return config

# # Define custom TransformerEncoder layer
# class TransformerEncoder(layers.Layer):
#     def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
#         super().__init__(**kwargs)
#         self.embed_dim = embed_dim
#         self.dense_dim = dense_dim
#         self.num_heads = num_heads
#         self.attention = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
#         self.dense_proj = keras.Sequential([
#             layers.Dense(dense_dim, activation="relu"),
#             layers.Dense(embed_dim)
#         ])
#         self.layernorm_1 = layers.LayerNormalization()
#         self.layernorm_2 = layers.LayerNormalization()

#     def call(self, inputs, mask=None):
#         attention_output = self.attention(inputs, inputs, attention_mask=mask)
#         proj_input = self.layernorm_1(inputs + attention_output)
#         proj_output = self.dense_proj(proj_input)
#         return self.layernorm_2(proj_input + proj_output)

#     def get_config(self):
#         config = super().get_config()
#         config.update({
#             "embed_dim": self.embed_dim,
#             "dense_dim": self.dense_dim,
#             "num_heads": self.num_heads,
#         })
#         return config

# # Load the dataset
# train_df = pd.read_csv(r'C:\Users\Tej Bachhav\OneDrive\Documents\NLP\Disaster\train_dataset.csv')

# # Concatenate "keyword", "location", and "text" columns while handling missing values
# train_df['combined_text'] = (train_df['keyword'].fillna('') + ' ' + 
#                               train_df['location'].fillna('') + ' ' + 
#                               train_df['text'].fillna(''))

# # Define constants
# max_length = 165
# max_tokens = 20_000
# batch_size = 32
# val_size = int(0.2 * len(train_df))

# # Create a tf.data.Dataset object from the combined text and target columns
# train_tf = tf.data.Dataset.from_tensor_slices(
#     (train_df['combined_text'], train_df['target'])
# )

# # Shuffle and batch the dataset
# train_tf = train_tf.shuffle(len(train_df)).batch(batch_size).prefetch(tf.data.AUTOTUNE)

# # Instantiate the TextVectorization layer
# text_vectorization = layers.TextVectorization(
#     max_tokens=max_tokens,
#     output_mode='int',
#     output_sequence_length=max_length
# )

# # Adapt the TextVectorization layer to learn the vocabulary from the combined text
# text_vectorization.adapt(train_tf.map(lambda twt, target: twt))

# # Map the vectorized text data back into the dataset
# train_tf = train_tf.map(lambda twt, target: (text_vectorization(twt), target), 
#                          num_parallel_calls=tf.data.AUTOTUNE)

# # Split the data into training and validation sets
# validation_data = train_tf.take(val_size).cache().prefetch(tf.data.AUTOTUNE)
# train_data = train_tf.skip(val_size).cache().prefetch(tf.data.AUTOTUNE)

# # Define the number of training steps per epoch
# steps_per_epoch = (len(train_df) - val_size) // batch_size
# validation_steps = val_size // batch_size

# # Define the inputs
# inputs = keras.Input(shape=(max_length,), dtype="int64")  # Input is tokenized text

# # Apply positional embeddings
# pos_embed = PositionalEmbedding(
#     sequence_length=max_length,   # Match your sequence length
#     input_dim=max_tokens,         # Match the vocabulary size
#     output_dim=256
# )(inputs)

# # Apply the Transformer encoder
# encoded = TransformerEncoder(embed_dim=256, dense_dim=32, num_heads=8)(pos_embed)

# # Apply global max pooling and dense layer for classification
# pooled_output = layers.GlobalMaxPooling1D()(encoded)
# final_output = layers.Dense(1, activation='sigmoid')(pooled_output)

# # Build the model
# model = keras.Model(inputs=inputs, outputs=final_output)

# # Compile the model
# model.compile(optimizer='adam', 
#               loss='binary_crossentropy', 
#               metrics=['accuracy'])

# # Print the model summary
# model.summary()

# # Define TensorBoard callback
# tensorboard_callback = keras.callbacks.TensorBoard(log_dir='logs', histogram_freq=1)

# # Train the model
# history = model.fit(
#     train_data,
#     epochs=10,
#     steps_per_epoch=steps_per_epoch,
#     validation_data=validation_data,
#     validation_steps=validation_steps,
#     callbacks=[
#         keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
#         tensorboard_callback  # Added TensorBoard callback
#     ]
# )

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Define custom PositionalEmbedding layer
class PositionalEmbedding(layers.Layer):
    def __init__(self, sequence_length, input_dim, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.token_embeddings = layers.Embedding(input_dim=input_dim, output_dim=output_dim)
        self.position_embeddings = layers.Embedding(input_dim=sequence_length, output_dim=output_dim)
        self.sequence_length = sequence_length
        self.input_dim = input_dim
        self.output_dim = output_dim

    def call(self, inputs):
        length = tf.shape(inputs)[-1]
        positions = tf.range(start=0, limit=length, delta=1)
        embedded_tokens = self.token_embeddings(inputs)
        embedded_positions = self.position_embeddings(positions)
        return embedded_tokens + embedded_positions

    def compute_mask(self, inputs, mask=None):
        if mask is None:
            return None
        return mask

    def get_config(self):
        config = super().get_config()
        config.update({
            "sequence_length": self.sequence_length,
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
        })
        return config

# Define custom TransformerEncoder layer
class TransformerEncoder(layers.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        self.attention = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.dense_proj = keras.Sequential(
            [layers.Dense(dense_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()

    def call(self, inputs, mask=None):
        attention_output = self.attention(inputs, inputs, attention_mask=mask)
        proj_input = self.layernorm_1(inputs + attention_output)
        proj_output = self.dense_proj(proj_input)
        return self.layernorm_2(proj_input + proj_output)

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "dense_dim": self.dense_dim,
            "num_heads": self.num_heads,
        })
        return config

