import tensorflow as tf
from tqdm import tqdm
import time, os

tf.enable_eager_execution()

class Model(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, units, rate, EPOCHS, df):
        super(Model, self).__init__()
        self.vocab_size = vocab_size
        self.units = units
        self.rate = rate
        self.EPOCHS = EPOCHS
        self.df = df

        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)

        if tf.test.is_gpu_available():
              self.gru = tf.keras.layers.CuDNNGRU(self.units, 
                                                  return_sequences=True, 
                                                  recurrent_initializer='glorot_uniform',
                                                  stateful=True)
        else:
              self.gru = tf.keras.layers.GRU(self.units, 
                                             return_sequences=True, 
                                             recurrent_activation='sigmoid', 
                                             recurrent_initializer='glorot_uniform', 
                                             stateful=True)
        
        self.dropout = tf.keras.layers.Dropout(self.rate)
        
        self.fc = tf.keras.layers.Dense(self.vocab_size)

        self.optimizer = tf.train.AdamOptimizer()


    # Using sparse_softmax_cross_entropy so that we don't have to create one-hot vectors
    def _loss_function(self, real, preds):
        return tf.losses.sparse_softmax_cross_entropy(labels=real, logits=preds)


    def forward(self, x):
        # output at every time step
        # output shape == (batch_size, seq_length, hidden_size) 

        embedding = self.embedding(x)
        output = self.gru(embedding)
        dropout_output = self.dropout(output)
        prediction = self.fc(dropout_output)
        # The dense layer will output predictions for every time_steps(seq_length)
        # output shape after the dense layer == (seq_length * batch_size, vocab_size)

        # states will be used to pass at every step to the model while training
        return prediction


    def train(self):
        print('Training...')

        # Directory where the checkpoints will be saved
        checkpoint_dir = './training_checkpoints'
        # Name of the checkpoint files
        checkpoint_prefix = os.path.join(checkpoint_dir, "word_generator")

        # Training loop
        for epoch in range(self.EPOCHS):
            start = time.time()
            
            # initializing the hidden state at the start of every epoch
            # initally hidden is None
            # hidden = self.reset_states()
            
            for (batch, (inp, target)) in tqdm(enumerate(self.df.dataset.take(10))):
                with tf.GradientTape() as tape:
                    # feeding the hidden state back into the model
                    # This is the interesting step
                    predictions = self.forward(inp)
                    loss = self._loss_function(target, predictions)
                    
                grads = tape.gradient(loss, self.variables)
                self.optimizer.apply_gradients(zip(grads, self.variables))

                if batch % 100 == 0:
                    print ('Epoch {} Batch {} Loss {:.4f}'.format(epoch+1,batch,loss))
            # saving (checkpoint) the model every 5 epochs
            if (epoch + 1) % 5 == 0:
                self.save_weights(checkpoint_prefix)

            print ('Epoch {} Loss {:.4f}'.format(epoch+1, loss))
            print ('Time taken for 1 epoch {} sec\n'.format(time.time() - start))
            print('Done!')

    def generate_text(self, start_string, num_generate):
        # Generating text using the learned model

        # Converting our start string to numbers (vectorizing) 
        input_eval = [self.df.word2idx[s] for s in [start_string] * self.df.BATCH_SIZE]
        input_eval = tf.expand_dims(input_eval, 1)

        # Empty string to store our results
        text_generated = []

        # Low temperatures results in more predictable text.
        # Higher temperatures results in more surprising text.
        # Experiment to find the best setting.
        temperature = 0.5

        # Evaluation loop.
        # Here batch size == 1
        # model.reset_states()
        for _ in range(num_generate):
            # print(input_eval)
            predictions = self.forward(input_eval)
            # remove the batch dimension
            predictions = tf.expand_dims(tf.squeeze(predictions[0], 0), 0)

            # using a multinomial distribution to predict the word returned by the model
            predictions = predictions / temperature
            predicted_id = tf.multinomial(predictions, num_samples=1)[-1,0].numpy()
            
            # We pass the predicted word as the next input to the model
            # along with the previous hidden state
            input_eval = tf.expand_dims([predicted_id] * self.df.BATCH_SIZE, 1)
            
            text_generated.append(self.df.idx2word[predicted_id])
            #if idx2word[predicted_id] == '.':
            #    break

        print (start_string + ' ' + ' '.join(text_generated))
