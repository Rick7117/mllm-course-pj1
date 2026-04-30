import os
import zipfile
import requests
from tqdm import tqdm
from config import COCO_IMAGES_DIR, COCO_ANNOTATIONS_DIR, DATA_DIR

def download_file(url, save_path):
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(save_path, 'wb') as f, tqdm(
        desc=os.path.basename(save_path),
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            bar.update(len(chunk))

def download_coco():
    images_url = "http://images.cocodataset.org/zips/val2017.zip"
    annotations_url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
    
    images_zip = os.path.join(DATA_DIR, 'val2017.zip')
    annotations_zip = os.path.join(DATA_DIR, 'annotations_trainval2017.zip')
    
    print("Downloading COCO val2017 images...")
    download_file(images_url, images_zip)
    
    print("Downloading COCO annotations...")
    download_file(annotations_url, annotations_zip)
    
    print("Extracting images...")
    with zipfile.ZipFile(images_zip, 'r') as zip_ref:
        zip_ref.extractall(DATA_DIR)
    
    print("Extracting annotations...")
    with zipfile.ZipFile(annotations_zip, 'r') as zip_ref:
        zip_ref.extractall(DATA_DIR)
    
    print("COCO dataset downloaded successfully!")

if __name__ == "__main__":
    download_coco()