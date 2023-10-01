import torch
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import numpy as np
import os
import time
import json
import requests
import subprocess
import zipfile
from tqdm import tqdm
from IPython.display import Image 


##automatically download the annotations and images (MS-COCO dataset) from COCO page
# Define the URLs for the annotations and image zip files
annotation_url = 'http://images.cocodataset.org/annotations/annotations_trainval2014.zip'
image_url = 'http://images.cocodataset.org/zips/val2014.zip'

# Define the destination directory for both zip files and extracted data
download_dir = os.path.abspath('.')
annotation_zip_path = os.path.join(download_dir, 'captions.zip')
image_zip_path = os.path.join(download_dir, 'val2014.zip')

# Use wget to download the annotations and image zip file
subprocess.run(['wget', annotation_url, '-O', annotation_zip_path])
subprocess.run(['wget', image_url, '-O', image_zip_path])

# Extract the annotations and image zip file
with zipfile.ZipFile(annotation_zip_path, 'r') as annotation_zip_ref:
    annotation_zip_ref.extractall(download_dir)

with zipfile.ZipFile(image_zip_path, 'r') as image_zip_ref:
    image_zip_ref.extractall(download_dir)


#remove the downloaded zip files if needed
if os.path.exists(annotation_zip_path):
    os.remove(annotation_zip_path)
if os.path.exists(image_zip_path):
    os.remove(image_zip_path)
    
# Define the path to the annotations file and image directory
annotation_file = os.path.join(download_dir, 'annotations', 'captions_val2014.json')
PATH = os.path.join(download_dir, 'val2014')

# Read the json file
with open(annotation_file, 'r') as ann_file:
    annotations = json.load(ann_file)
    
# Load the senticap annotations
with open("senticap_dataset.json", 'r') as f:
    senti_annotations = json.load(f)

# List to maintain the index of captions and image name paths
all_captions = []
all_img_name_vector = []
sentiment = []

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
    
# iterate on the annotations of COCO validation captions
for annot in annotations['annotations']:
    image_id = annot['image_id']
    if image_id in res:
        caption = '<start> ' + annot['caption'] + ' <end>'
        full_coco_image_path = PATH + 'COCO_val2014_' + '%012d.jpg' % (image_id)
        all_img_name_vector.append([full_coco_image_path,0])
        all_captions.append(caption)