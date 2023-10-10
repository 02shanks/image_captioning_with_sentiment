import os
import json
from sklearn.utils import shuffle
import tensorflow as tf
from sklearn.model_selection import train_test_split
from utils import calc_max_length, map_func

def download_datastet():
    annotation_zip = tf.keras.utils.get_file('captions.zip',
                                          cache_subdir=os.path.abspath('.'),
                                          origin = 'http://images.cocodataset.org/annotations/annotations_trainval2014.zip',
                                          extract = True)
    annotation_file = os.path.dirname(annotation_zip)+'/annotations/captions_val2014.json'

    name_of_zip = 'val2014.zip'
    if not os.path.exists(os.path.abspath('.') + '/' + name_of_zip):
        image_zip = tf.keras.utils.get_file(name_of_zip,
                                            cache_subdir=os.path.abspath('.'),
                                            origin = 'http://images.cocodataset.org/zips/val2014.zip',
                                            extract = True)
        PATH = os.path.dirname(image_zip)+'/val2014/'
    else:
        PATH = os.path.abspath('.')+'/val2014/'
        
        # Read the json file
    with open(annotation_file, 'r') as ann_file:
        annotations = json.load(ann_file)

    # List to maintain the index of captions and image name paths
    all_captions = []
    all_img_name_vector = []

    # Load the senticap annotations
    with open("senticap_dataset.json", 'r') as f:
        senti_annotations = json.load(f)

    # form a similar file asthat of the validation captions of COCO
    senti_data = {"annotations":[]}
    for annot in senti_annotations['images']:
        img_id = int(annot['filename'].split('_')[-1].split('.')[0])
        for sen in annot['sentences']:
            senti_data["annotations"].append({"image_id":img_id,"caption":sen['raw'],"sentiment":sen["sentiment"]}) 

    # collect all the uniques image_id
    res = [v['image_id'] for v in senti_data['annotations']]
    res = set(res)

    # iterate on the annotations
    for entry in senti_data['annotations']:
        caption = '<start> ' + entry['caption'] + ' <end>'
        all_captions.append(caption)
        img_id = entry['image_id']
        full_coco_image_path = PATH + 'COCO_val2014_' + '%012d.jpg' % (img_id)
        senti = entry['sentiment']
        if senti == 0:
            senti = -1
        all_img_name_vector.append([full_coco_image_path,senti])
        
    # iterate on the annotations of COCO calidation captions
    for annot in annotations['annotations']:
        image_id = annot['image_id']
        if image_id in res:
            caption = '<start> ' + annot['caption'] + ' <end>'
            full_coco_image_path = PATH + 'COCO_val2014_' + '%012d.jpg' % (image_id)
            all_img_name_vector.append([full_coco_image_path,0])
            all_captions.append(caption)
            

    return all_captions, all_img_name_vector



def create_dataset(train_captions, sentiment):
    # Choose the top 5000 words from the vocabulary
    top_k = 5000
    # we tokenize the captions to only include top 5000 words, we exclude other infrequent words and replace them with <unk>, we also remove the punctuations 
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=top_k,
                                                    oov_token="<unk>",
                                                    filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')
    tokenizer.fit_on_texts(train_captions)
    #text to sequence expects a sequence of words
    train_seqs = tokenizer.texts_to_sequences(train_captions)

    # we change <pad> which is default word based on the longest caption with 0
    tokenizer.word_index['<pad>'] = 0
    tokenizer.index_word[0] = '<pad>'

    # Create the tokenized vectors
    train_seqs = tokenizer.texts_to_sequences(train_captions)
    cap_vector = tf.keras.preprocessing.sequence.pad_sequences(train_seqs, padding='post')
    max_length = calc_max_length(train_seqs)

    # We use the train test split to split 80%-20% data of train itself
    # here we split the sentiment data to maintain the sequence of pfile path caption and sentiment
    img_name_train, img_name_val, cap_train, cap_val = train_test_split(sentiment,
                                                                        cap_vector,
                                                                        test_size=0.2,
                                                                        random_state=0)
    
    
    # This batch size should be perfectly divisible to the num of examples that are considered
    BATCH_SIZE = 100
    BUFFER_SIZE = 1000

    dataset = tf.data.Dataset.from_tensor_slices(([x[0] for x in img_name_train], cap_train,[x[1] for x in img_name_train]))

    # Use map to load the numpy files in parallel
    dataset = dataset.map(lambda item1, item2, item3: tf.numpy_function(
            map_func, [item1,item2,item3], [tf.float32, tf.int32,tf.int32]),
            num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # We sorted the encode train to store the numpy files hence we shuffle the dataset pair back according to batch size and form batches
    dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    
    return dataset, tokenizer, img_name_train, img_name_val, cap_train, cap_val, max_length


