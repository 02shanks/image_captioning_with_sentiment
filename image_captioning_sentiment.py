import torch
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import numpy as np
import os
import time
import json
import requests
import zipfile
from tqdm import tqdm
from IPython.display import Image 


##automatically download the annotations and images (MS-COCO dataset) from COCO page
import os
import requests
import zipfile

# Define the URL for the annotations and image zip file
annotation_url = 'http://images.cocodataset.org/annotations/annotations_trainval2014.zip'
image_url = 'http://images.cocodataset.org/zips/val2014.zip'

# Define the destination directory for both zip files and extracted data
download_dir = os.path.abspath('.')
annotation_zip_path = os.path.join(download_dir, 'captions.zip')
image_zip_path = os.path.join(download_dir, 'val2014.zip')

# Download the annotations zip file
annotation_response = requests.get(annotation_url)
with open(annotation_zip_path, 'wb') as annotation_zip_file:
    annotation_zip_file.write(annotation_response.content)

# Extract the annotations zip file
with zipfile.ZipFile(annotation_zip_path, 'r') as annotation_zip_ref:
    annotation_zip_ref.extractall(download_dir)

# Define the path to the annotations file
annotation_file = os.path.join(download_dir, 'annotations', 'captions_val2014.json')

# Check if the image zip file exists and download it if necessary
if not os.path.exists(image_zip_path):
    image_response = requests.get(image_url)
    with open(image_zip_path, 'wb') as image_zip_file:
        image_zip_file.write(image_response.content)

# Define the path to the image directory
image_dir = os.path.join(download_dir, 'val2014')

# Clean up - remove the downloaded zip files if needed
if os.path.exists(annotation_zip_path):
    os.remove(annotation_zip_path)
if os.path.exists(image_zip_path):
    os.remove(image_zip_path)
