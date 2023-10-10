from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
# import for splitting train test data
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import numpy as np
import os
import time
import json
# package to get a progress bar for saving vectors of each image in a .npy file
from tqdm import tqdm
from IPython.display import Image 
from dataset import download_datastet, create_dataset
from model import extract_save_feature, CNN_Encoder, RNN_Decoder, train_step, load_model
from utils import calc_max_length, evaluate

all_captions, all_img_name_vector = download_datastet()

# shuffle the arrays to avoid grouping of different captions with same images
train_captions, img_name_vector = shuffle(all_captions,
                                        all_img_name_vector,
                                        random_state=1)

# Select the first 20000 captions from the shuffled set out of 20002
num_examples = 20000
train_captions = train_captions[:num_examples]
img_name_vector = img_name_vector[:num_examples]
sentiment = img_name_vector

extract_save_feature(img_name_vector)
dataset, tokenizer, img_name_train, img_name_val, cap_train, cap_val, max_length = create_dataset(train_captions, sentiment)
# call to encoder and decoder classes
# This batch size should be perfectly divisible to the num of examples that are considered
BATCH_SIZE = 100
BUFFER_SIZE = 1000
# The vector embeddign size for deciding the RNN_GRU decoder 
embedding_dim = 256

# number of GRU units in the cell
units = 512

# Total number of terms in the vocabulary - 6409 total vocab_size
vocab_size = len(tokenizer.word_index) + 1

# total 20000 images hence the num_steps will be 200
num_steps = len(img_name_train) // BATCH_SIZE
# Shape of the vector extracted from InceptionV3 is (64, 2052)
# These two variables represent that vector shape
features_shape = 2052
attention_features_shape = 64
encoder = CNN_Encoder(embedding_dim)
decoder = RNN_Decoder(embedding_dim, units, vocab_size)

# set optimizer
optimizer = tf.keras.optimizers.Adam()

# set up of checkpoint and checkpoint manager
checkpoint_path = "./checkpoints/sentiment"
ckpt = tf.train.Checkpoint(encoder=encoder,
                           decoder=decoder,
                           optimizer = optimizer)
ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

start_epoch = 0
if ckpt_manager.latest_checkpoint:
  start_epoch = int(ckpt_manager.latest_checkpoint.split('-')[-1])
  
  
EPOCHS = 100

for epoch in range(start_epoch, EPOCHS):
    start = time.time()
    total_loss = 0
    # dataset will have 3 parameters tensor and caption,sentiment and batches
    for (batch, (img_tensor, target,senti)) in enumerate(dataset):
        batch_loss, t_loss = train_step(img_tensor, target, decoder,tokenizer,encoder,optimizer,BATCH_SIZE)
        total_loss += t_loss

        if batch % 100 == 0:
            print ('Epoch {} Batch {} Loss {:.4f}'.format(
              epoch + 1, batch, batch_loss.numpy() / int(target.shape[1])))
    # storing the epoch end loss value to plot later
    if epoch % 5 == 0:
      ckpt_manager.save()

    print ('Epoch {} Loss {:.6f}'.format(epoch + 1,
                                         total_loss/num_steps))
    print ('Time taken for 1 epoch {} sec\n'.format(time.time() - start))


# captions on the validation set
rid = np.random.randint(0, len(img_name_val))
image = img_name_val[rid]
real_caption = ' '.join([tokenizer.index_word[i] for i in cap_val[rid] if i not in [0]])
image_features_extract_model = load_model()
result = evaluate(image, encoder, decoder, image_features_extract_model, tokenizer, max_length)

print ('Real Caption:', real_caption)
print ('Prediction Caption:', ' '.join(result))
print("Real caption's sentiment is:",image[1])
# opening the image
pil_img = Image(image[0])
display(pil_img)