import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

# 读取语料库
with open('corpus.txt', 'r', encoding='utf-8') as f:
    text_data = f.read()

# 分句
lines = text_data.split('\n')

# 使用Tokenizer进行分词
word_tokenizer = Tokenizer()
word_tokenizer.fit_on_texts(lines)
vocab_size = len(word_tokenizer.word_index) + 1

# 创建输入和输出序列
sequences = []
for line in lines:
    token_list = word_tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        sequences.append(n_gram_sequence)

# 填充序列
max_seq_length = max([len(seq) for seq in sequences])
sequences = np.array(pad_sequences(sequences, maxlen=max_seq_length, padding='pre'))

# 创建训练数据
input_data, target_data = sequences[:,:-1], sequences[:,-1]
target_data = tf.keras.utils.to_categorical(target_data, num_classes=vocab_size)

# 定义模型
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, 128, input_length=max_seq_length-1),
    tf.keras.layers.Bidirectional(tf.keras.layers.GRU(64, return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.GRU(64)),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(vocab_size, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(input_data, target_data, epochs=100, verbose=1)

# 文本生成函数
def generate_text(seed_text, num_words, trained_model, max_len):
    for _ in range(num_words):
        token_list = word_tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_len-1, padding='pre')
        predicted = np.argmax(trained_model.predict(token_list), axis=-1)
        predicted_word = word_tokenizer.index_word[predicted[0]]
        seed_text += " " + predicted_word
    return seed_text

# 生成文本
print(generate_text("段誉心花怒放，抱著她身子一跃而起，“啊哈”一声，拍的一声响，重又落入污泥之中，伸嘴过去，便要吻她樱唇。王语嫣宛转相就，四唇正欲相接，突然间头顶呼呼风响，甚麽东西落将下来。
　　两人吃了一惊，忙向井栏2边一靠，砰的一声响，有人落入井中。", 100, model, max_seq_length))
