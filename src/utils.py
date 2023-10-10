import numpy as np
import tensorflow as tf

# The maximum size of captions to take the padding limit 
def calc_max_length(tensor):
    return max(len(t) for t in tensor)

def load_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    # inception v3 requires all the images in a 299*299 pixel format
    img = tf.image.resize(img, (299, 299))
    #using the preprocess_input function of the inception_v3 
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    return img, image_path

# Load the numpy files each numpy file has a shape of 64x2052
def map_func(img_name, cap,senti):
  img_tensor = np.load(img_name.decode('utf-8')+'.npy')
  # after loading the numpy files we make a similar vector with 4 more dimensions at axis 1
  comb_feats = np.zeros([img_tensor.shape[0],img_tensor.shape[1]+4], dtype=np.float32)
  comb_feats[:,:2048] = img_tensor
  # adding an encoding vale which corresponds to sentiments
  #Positive sentiment
  if senti == 1:
      result = [1,0,0,1]
  # negative sentiment
  elif senti == -1:
      result = [0,0,1,2]
  # neutral sentiment
  else:
      result = [0,1,0,0]
  comb_feats[:,2048:] = result
  # combine and return the new image vector of shape 64x2052
  return comb_feats,cap,senti


def evaluate(image, encoder, decoder, image_features_extract_model, tokenizer, max_length):
    # decoder reset state with start tag
    hidden = decoder.reset_state(batch_size=1)

    # Extract the image to be tested
    temp_input = tf.expand_dims(load_image(image[0])[0], 0)
    img_tensor_val = image_features_extract_model(temp_input)
    img_tensor_val = tf.reshape(img_tensor_val, (img_tensor_val.shape[0], -1, img_tensor_val.shape[3]))
    new_img_tensor_val = np.zeros([img_tensor_val.shape[0],img_tensor_val.shape[1],(img_tensor_val.shape[2]+4)],dtype=np.float32)
    # Only the first 2048 vectors will be returned
    # Give user input to decide the level of sentiment we need from the image
    # 0 for neutral, 1 for positive, -1 for negative
    new_img_tensor_val[:,:,:2048] = img_tensor_val
    if image[1] == 0:
        new_img_tensor_val[:,:,2048:] = [0,1,0,0]
    elif image[1] == 1:
        new_img_tensor_val[:,:,2048:] = [1,0,0,1]
    elif image[1] == -1:
        new_img_tensor_val[:,:,2048:] = [0,0,1,2]
    
    # form new image tensor 
    img_tensor_val = new_img_tensor_val
        
    # pass it through encoder
    features = encoder(img_tensor_val)

    # start token by default to all the captions in batch
    dec_input = tf.expand_dims([tokenizer.word_index['<start>']], 0)
    result = []
    
    
    # keep generating the caption till max length
    for i in range(max_length):
        predictions, hidden, attention_weights = decoder(dec_input, features, hidden)

        predicted_id = tf.argmax(predictions[0]).numpy()
        result.append(tokenizer.index_word[predicted_id])

        if tokenizer.index_word[predicted_id] == '<end>':
            return result

        dec_input = tf.expand_dims([predicted_id], 0)

    return result