from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import numpy as np
import os
import time
import json
import requests
import subprocess
import zipfile
from PIL import Image
from tqdm import tqdm

import torch
import torchvision.models as models
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset


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
        
# shuffle the arrays to avoid grouping of different captions with same images
train_captions, img_name_vector = shuffle(all_captions,
                                          all_img_name_vector,
                                          random_state=1)

# Select the first 20000 captions from the shuffled set out of 20002
num_examples = 20000
train_captions = train_captions[:num_examples]
img_name_vector = img_name_vector[:num_examples]

# Load the Inception V3 model pre-trained on ImageNet
image_model = models.inception_v3(pretrained=True)

# Remove the top classification layer (include_top=False)
modules = list(image_model.children())[:-1]  # Remove the last layer

# Create a new PyTorch model with the modified architecture
image_features_extract_model = torch.nn.Sequential(*modules)

# Set the model to evaluation mode (no gradient calculation)
image_features_extract_model.eval()


# img_name_vector has a file path and sentiment association
sentiment = img_name_vector
# remove sentiment after creating a copy
img_name_vector = [x[0] for x in img_name_vector]


# Define a function to load an image and apply transformations
def load_image(image_path):
    img = Image.open(image_path)
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img = transform(img)
    return img, image_path

# Define a custom dataset class
class CustomDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx]
        img, _ = load_image(img_path)
        if self.transform:
            img = self.transform(img)
        return img, img_path

# Define a function to process and save feature vectors
def save_features(image_data_loader, model, output_dir):
    for img_batch, path_batch in tqdm(image_data_loader):
        # We call the model to get the feature vectors
        batch_features = model(img_batch)

        # Reshape the feature vectors
        batch_features = batch_features.view(batch_features.size(0), -1, batch_features.size(3))

        for bf, p in zip(batch_features, path_batch):
            # Save the feature vectors as numpy files
            path_of_feature = p
            np.save(os.path.join(output_dir, os.path.basename(path_of_feature) + '.npy'), bf.detach().cpu().numpy())

# Define the paths to image files
encode_train = sorted(set(img_name_vector))

# Create a custom dataset from the image file paths
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
image_dataset = CustomDataset(encode_train, transform=transform)

# Create a DataLoader for batching the dataset
batch_size = 16
image_data_loader = DataLoader(image_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

# Define the directory to save feature vectors
output_dir = 'feature_vectors'

# Make sure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Set the model to evaluation mode (no gradient calculation)
image_features_extract_model.eval()

# Save the feature vectors
save_features(image_data_loader, image_features_extract_model, output_dir)
