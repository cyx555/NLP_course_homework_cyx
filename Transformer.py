import os
import time
import tensorflow as tf
import numpy as np
from datasets import Dataset
from sklearn.model_selection import train_test_split
import re

start_time = time.time()

data_path = './data'
file_list = [os.path.join(data_path, f) for f in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, f))]

corpus = []
for file in file_list:
    try:
        with open(file, encoding='ansi', errors='ignore') as f:
            corpus.append(f.read())
    except Exception as e:
        print(f"Error reading {file}: {e}")

# Create dataset
data = Dataset.from_dict({"text": corpus})

# Custom Tokenizer
class TextTokenizer:
    def __init__(self, corpus):
        self.vocab = self.build_vocab(corpus)
        self.vocab_size = len(self.vocab)
        self.token_to_id = {token: idx for idx, token in enumerate(self.vocab)}
        self.id_to_token = {idx: token for token, idx in self.token_to_id.items()}

    def build_vocab(self, corpus):
        tokens = set()
        for text in corpus:
            tokens.update(re.findall(r'\w+|\S', text))
        return sorted(tokens)

    def encode(self, text):
        return [self.token_to_id[token] for token in re.findall(r'\w+|\S', text)]

    def decode(self, token_ids):
        return ''.join([self.id_to_token[token_id] for token_id in token_ids])

# Initialize tokenizer
tokenizer = TextTokenizer(corpus)

# Tokenize dataset
def tokenize_examples(examples):
    return {"input_ids": [tokenizer.encode(text) for text in examples["text"]]}

tokenized_data = data.map(tokenize_examples, batched=True, remove_columns=["text"])


max_seq_len = 512
padded_input_ids = tf.keras.preprocessing.sequence.pad_sequences(
    [x for x in tokenized_data['input_ids']], maxlen=max_seq_len, padding='post'
)

train_ids, val_ids = train_test_split(padded_input_ids, test_size=0.1)

train_ds = tf.data.Dataset.from_tensor_slices((train_ids, train_ids)).shuffle(len(train_ids)).batch(4)
val_ds = tf.data.Dataset.from_tensor_slices((val_ids, val_ids)).batch(4)

class CustomTransformer(tf.keras.Model):
    def __init__(self, vocab_size, d_model=512, num_heads=8, num_layers=6, dim_ff=2048, max_seq_len=512):
        super(CustomTransformer, self).__init__()
        self.embedding = tf.keras.layers.Embedding(vocab_size, d_model)
        self.pos_encoding = self.positional_encoding(max_seq_len, d_model)
        self.encoder_layers = [
            tf.keras.layers.LayerNormalization(epsilon=1e-6) for _ in range(num_layers)
        ]
        self.encoder_layers += [
            tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model//num_heads) for _ in range(num_layers)
        ]
        self.dense_ff = tf.keras.layers.Dense(dim_ff, activation='relu')
        self.output_layer = tf.keras.layers.Dense(vocab_size)

    def call(self, inputs, training=False, tgt=None):
        x = self.embedding(inputs[0]) + self.pos_encoding
        for ln, mha in zip(self.encoder_layers[:len(self.encoder_layers)//2], self.encoder_layers[len(self.encoder_layers)//2:]):
            x = ln(x)
            x = mha(x, x, return_attention_scores=False, training=training)
        x = self.dense_ff(x)
        return self.output_layer(x)

    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(np.arange(position)[:, np.newaxis], np.arange(d_model)[np.newaxis, :], d_model)
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        pos_encoding = angle_rads[np.newaxis, ...]
        return tf.cast(pos_encoding, dtype=tf.float32)

    def get_angles(self, pos, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        return pos * angle_rates

# Instantiate model
vocab_size = tokenizer.vocab_size
model = CustomTransformer(vocab_size)

# Compile model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

# Training with device assignment
with tf.device('/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'):
    model.fit(
        train_ds, validation_data=val_ds, epochs=3,
        callbacks=[EarlyStopping(patience=2, restore_best_weights=True),
                   ModelCheckpoint(filepath='./transformer_checkpoint', save_best_only=True)]
    )

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Training completed in: {elapsed_time // 3600}h {(elapsed_time % 3600) // 60}m {elapsed_time % 60}s")

model.save_weights("./transformer_novels_weights.h5")
np.save("./transformer_novels_vocab.npy", tokenizer.vocab)

# Text generation
def generate_text(model, tokenizer, seed_text, max_len=200):
    input_ids = tokenizer.encode(seed_text)
    generated_ids = input_ids
    for _ in range(max_len):
        output = model(tf.constant([generated_ids]), tf.constant([generated_ids]))
        next_token_id = tf.argmax(output[0, -1, :])
        generated_ids.append(next_token_id.numpy())
        if next_token_id == tokenizer.token_to_id.get('[SEP]', -1):
            break
    return tokenizer.decode(generated_ids)

seed_text = "段誉心花怒放，抱著她身子一跃而起，“啊哈”一声，拍的一声响，重又落入污泥之中，伸嘴过去，便要吻她樱唇。王语嫣宛转相就，四唇正欲相接，突然间头顶呼呼风响，甚麽东西落将下来。两人吃了一惊，忙向井栏2边一靠，砰的一声响，有人落入井中。"
generated_text = generate_text(model, tokenizer, seed_text)
