import numpy as np
import tensorflow as tf
from tqdm import tqdm
from sklearn.utils import shuffle
from dataset import download_datastet
from utils import load_image, calc_max_length

def load_model():
    image_model = tf.keras.applications.InceptionV3(include_top=False,
                                                    weights='imagenet')
    # this gets the second last vector
    new_input = image_model.input
    hidden_layer = image_model.layers[-1].output

    image_features_extract_model = tf.keras.Model(new_input, hidden_layer)
    
    return image_features_extract_model

def extract_save_feature(img_name_vector):

    image_features_extract_model = load_model()
    # remove sentiment after creating a copy
    img_name_vector = [x[0] for x in img_name_vector]

    encode_train = sorted(set(img_name_vector))

    # Feel free to change batch_size according to your system configuration
    image_dataset = tf.data.Dataset.from_tensor_slices(encode_train)
    # parallel execution to convert each image Inceptionv3 compatible format
    image_dataset = image_dataset.map(
    load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(16)


    for img, path in tqdm(image_dataset):
        # We call the model function to get the vector of shape 81x2048 
        batch_features = image_features_extract_model(img)
        
        batch_features = tf.reshape(batch_features,
                                    (batch_features.shape[0], -1, batch_features.shape[3]))

        for bf, p in zip(batch_features, path):
            # save the vectors as numpy files with the .npy extension by default  
            path_of_feature = p.numpy().decode("utf-8")
            np.save(path_of_feature, bf.numpy())
            
            
class BahdanauAttention(tf.keras.Model):
  def __init__(self, units):
    super(BahdanauAttention, self).__init__()
    self.W1 = tf.keras.layers.Dense(units)
    self.W2 = tf.keras.layers.Dense(units)
    self.V = tf.keras.layers.Dense(1)

  def call(self, features, hidden):
    # hidden_with_time_axis shape is batch_size, 1, hidden_size
    hidden_with_time_axis = tf.expand_dims(hidden, 1)

    # score shape is batch_size, 64, hidden_size
    score = tf.nn.tanh(self.W1(features) + self.W2(hidden_with_time_axis))

    # score shape is batch_size, 64, hidden_size
    attention_weights = tf.nn.softmax(self.V(score), axis=1)

    # context_vector is batch_size, hidden_size
    context_vector = attention_weights * features
    context_vector = tf.reduce_sum(context_vector, axis=1)

    return context_vector, attention_weights

class CNN_Encoder(tf.keras.Model):
    # Since you have already extracted the features and dumped it using pickle
    # This encoder passes those features through a Fully connected layer
    def __init__(self, embedding_dim):
        super(CNN_Encoder, self).__init__()
        # input shape is batchx64x2052 while output is batchx64x256(embedding dimensions) with a dense layer
        self.fc = tf.keras.layers.Dense(embedding_dim)

    def call(self, x):
       # the input to the dense layer will be the image tensor
        x = self.fc(x)
        # relu activation over the fully connected maintains the shape
        x = tf.nn.relu(x)
        return x
    
class RNN_Decoder(tf.keras.Model):
  def __init__(self, embedding_dim, units, vocab_size):
    super(RNN_Decoder, self).__init__()
    # units in the RNN GRU cell = 512
    self.units = units

    # define Embedding layer with the vocab_size and embedding dimension 6409 and 256
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    # define GRU cell with 256 units with return sequences set to true to give successive outputs to next unit
    self.gru = tf.keras.layers.GRU(self.units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')
    # dense layers with output dim 512
    self.fc1 = tf.keras.layers.Dense(self.units)
    # dense layer with 6409 dimensions to get back the token numbers activated
    self.fc2 = tf.keras.layers.Dense(vocab_size)

    # Call to define Attention object
    self.attention = BahdanauAttention(self.units)

  def call(self, x, features, hidden):
    # defining attention as a separate model
    context_vector, attention_weights = self.attention(features, hidden)

    # the shape will be batch_sizex1x256
    x = self.embedding(x)

    # shape will be batch_sizex1x(256+hiddensize)
    x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

    #pass the concatenated output to GRU cell
    output, state = self.gru(x)

    # shape will be batch_size x max_length x hidden_size
    x = self.fc1(output)

    # collapsing 1st dimension
    x = tf.reshape(x, (-1, x.shape[2]))

    # output shape batch_size * max_length, vocab
    x = self.fc2(x)

    return x, state, attention_weights

  def reset_state(self, batch_size):
    return tf.zeros((batch_size, self.units))


def loss_function(real, pred):
  loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
  mask = tf.math.logical_not(tf.math.equal(real, 0))
  loss_ = loss_object(real, pred)

  mask = tf.cast(mask, dtype=loss_.dtype)
  loss_ *= mask

  return tf.reduce_mean(loss_)


@tf.function
def train_step(img_tensor, target, decoder,tokenizer,encoder,optimizer,BATCH_SIZE):
  loss = 0
  # set initial zeros with start tokens of 100 or batch_size
  hidden = decoder.reset_state(batch_size=target.shape[0])

  # Every time the dec_input will have the previous word predicted going as input
  dec_input = tf.expand_dims([tokenizer.word_index['<start>']] * BATCH_SIZE, 1)
 
  with tf.GradientTape() as tape:
      features = encoder(img_tensor)

      # pass the image tensor through the encoder for embedding dimension based output
      for i in range(1, target.shape[1]):
          # passing the features through the decoder
          predictions, hidden, _ = decoder(dec_input, features, hidden)

          # calculate loss based on the current word predicted
          loss += loss_function(target[:, i], predictions)

          # using the predicted word as next unit input
          dec_input = tf.expand_dims(target[:, i], 1)

  # calculate total loss based on number of words in caption
  total_loss = (loss / int(target.shape[1]))

  # get all trainable variables that can be optimized with back prop
  trainable_variables = encoder.trainable_variables + decoder.trainable_variables

  gradients = tape.gradient(loss, trainable_variables)
  # apply weight optimization on trainable variables
  optimizer.apply_gradients(zip(gradients, trainable_variables))

  return loss, total_loss