import tensorflow as tf
import numpy as np

class Dataset:
    def __init__(self, seq_length, BATCH_SIZE = 40, BUFFER_SIZE = 10000):
        self.seq_length = seq_length
        self.BATCH_SIZE = BATCH_SIZE
        self.BUFFER_SIZE = BUFFER_SIZE
        self.word2idx = {}
        self.idx2word = np.array([])
        self.vocab = set()
        self.dataset = None

    def make(self, text_words):
        def split_input_target(chunk):
            input_text = chunk[:-1]
            target_text = chunk[1:]
            return input_text, target_text

        self.vocab = sorted(set(text_words))
        self.word2idx = {u: v for v, u in enumerate(self.vocab)}
        self.idx2word = np.array(self.vocab)
        text_as_int = np.array([self.word2idx[c] for c in text_words])

        chunks = tf.data.Dataset.from_tensor_slices(text_as_int).batch(self.seq_length+1, drop_remainder=True)
        self.dataset = chunks.map(split_input_target)
        self.dataset = self.dataset.shuffle(self.BUFFER_SIZE).batch(self.BATCH_SIZE, drop_remainder=True)
