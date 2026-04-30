import json
import os
from config import COCO_IMAGES_DIR, COCO_ANNOTATIONS_DIR

class COCODataset:
    def __init__(self):
        self.annotations_file = os.path.join(COCO_ANNOTATIONS_DIR, 'captions_val2017.json')
        self.images_dir = COCO_IMAGES_DIR
        self.load_annotations()
    
    def load_annotations(self):
        with open(self.annotations_file, 'r') as f:
            self.data = json.load(f)
        
        self.image_id_to_filename = {}
        for image in self.data['images']:
            self.image_id_to_filename[image['id']] = image['file_name']
        
        self.annotations = {}
        for ann in self.data['annotations']:
            image_id = ann['image_id']
            if image_id not in self.annotations:
                self.annotations[image_id] = []
            self.annotations[image_id].append(ann['caption'])
    
    def get_image_path(self, image_id):
        filename = self.image_id_to_filename.get(image_id)
        if filename:
            return os.path.join(self.images_dir, filename)
        return None
    
    def get_captions(self, image_id):
        return self.annotations.get(image_id, [])
    
    def get_all_image_ids(self):
        return list(self.image_id_to_filename.keys())
    
    def get_image_caption_pairs(self, max_samples=None):
        pairs = []
        for image_id in self.get_all_image_ids():
            captions = self.get_captions(image_id)
            if captions:
                pairs.append({
                    'image_id': image_id,
                    'image_path': self.get_image_path(image_id),
                    'caption': captions[0]
                })
            if max_samples and len(pairs) >= max_samples:
                break
        return pairs