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
image_dir = os.path.join(download_dir, 'val2014')