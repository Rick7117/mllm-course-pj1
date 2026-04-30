import numpy as np
import os
import pickle
from tqdm import tqdm
from config import EMBEDDINGS_DIR, RESULTS_DIR
from dataset import COCODataset
from models import CLIPModel, BLIPModel, BLIP2Model
from evaluator import RetrievalEvaluator

def extract_embeddings(model, dataset, model_name, max_samples=1000):
    cache_path = os.path.join(EMBEDDINGS_DIR, f'{model_name}_embeddings.pkl')
    
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            return pickle.load(f)
    
    image_ids = dataset.get_all_image_ids()[:max_samples]
    
    image_features = []
    text_features = []
    image_to_text_mapping = {}
    captions = []
    
    text_idx = 0
    for i, image_id in enumerate(tqdm(image_ids, desc=f"Extracting {model_name} features")):
        image_path = dataset.get_image_path(image_id)
        caption_list = dataset.get_captions(image_id)
        
        if image_path and caption_list:
            img_feat = model.extract_image_features(image_path)
            image_features.append(img_feat)
            
            caption = caption_list[0]
            txt_feat = model.extract_text_features(caption)
            text_features.append(txt_feat)
            captions.append(caption)
            image_to_text_mapping[i] = text_idx
            text_idx += 1
    
    result = {
        'image_features': np.array(image_features),
        'text_features': np.array(text_features),
        'image_to_text_mapping': image_to_text_mapping,
        'captions': captions
    }
    
    with open(cache_path, 'wb') as f:
        pickle.dump(result, f)
    
    return result

def run_retrieval():
    dataset = COCODataset()
    
    models = {
        'CLIP': CLIPModel(),
        'BLIP': BLIPModel(),
        'BLIP2': BLIP2Model()
    }
    
    results = {}
    
    for model_name, model in models.items():
        print(f"\nProcessing {model_name}...")
        
        data = extract_embeddings(model, dataset, model_name, max_samples=500)
        
        image_features = data['image_features']
        text_features = data['text_features']
        image_to_text_mapping = data['image_to_text_mapping']
        
        print(f"Image features shape: {image_features.shape}")
        print(f"Text features shape: {text_features.shape}")
        
        it_recall, ti_recall = RetrievalEvaluator.compute_recall_at_k(
            image_features, text_features, image_to_text_mapping, k_list=[1, 5, 10]
        )
        
        results[model_name] = {
            'Image-to-Text': it_recall,
            'Text-to-Image': ti_recall
        }
        
        print(f"Image-to-Text Recall@1: {it_recall[1]:.4f}, Recall@5: {it_recall[5]:.4f}, Recall@10: {it_recall[10]:.4f}")
        print(f"Text-to-Image Recall@1: {ti_recall[1]:.4f}, Recall@5: {ti_recall[5]:.4f}, Recall@10: {ti_recall[10]:.4f}")
    
    with open(os.path.join(RESULTS_DIR, 'retrieval_results.json'), 'w') as f:
        import json
        json.dump(results, f, indent=2)
    
    print("\nResults saved to retrieval_results.json")

if __name__ == "__main__":
    run_retrieval()