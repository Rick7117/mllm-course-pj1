import json
import os
import pickle

import numpy as np
from scipy.spatial.distance import cdist
from tqdm import tqdm

from config import EMBEDDINGS_DIR, NUM_NEIGHBORS, NUM_QUALITATIVE_EXAMPLES, NUM_SAMPLES_NEAREST_NEIGHBOR, RESULTS_DIR
from dataset import COCODataset
from models import BLIP2Model, BLIPModel, CLIPModel


def normalize_features(features):
    norms = np.linalg.norm(features, axis=-1, keepdims=True)
    return features / np.clip(norms, 1e-12, None)


def extract_nn_embeddings(model, dataset, model_name, num_samples=1000):
    cache_name = getattr(model, "cache_name", model_name.lower())
    cache_path = os.path.join(EMBEDDINGS_DIR, f"{cache_name}_{num_samples}_nn_embeddings.pkl")

    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    pairs = dataset.get_image_caption_pairs(max_samples=num_samples)

    image_features = []
    text_features = []
    captions = []
    image_paths = []
    image_ids = []

    for pair in tqdm(pairs, desc=f"Extracting {model_name} NN features"):
        img_feat = model.extract_image_features(pair["image_path"])
        txt_feat = model.extract_text_features(pair["caption"])

        image_features.append(img_feat)
        text_features.append(txt_feat)
        captions.append(pair["caption"])
        image_paths.append(pair["image_path"])
        image_ids.append(pair["image_id"])

    result = {
        "image_features": np.array(image_features),
        "text_features": np.array(text_features),
        "captions": captions,
        "image_paths": image_paths,
        "image_ids": image_ids,
    }

    with open(cache_path, "wb") as f:
        pickle.dump(result, f)

    return result


def find_nearest_neighbors(query_features, target_features, k=5):
    query_features = normalize_features(query_features)
    target_features = normalize_features(target_features)

    if query_features.ndim == 3 and target_features.ndim == 2:
        similarity = np.einsum("nqd,md->nqm", query_features, target_features).max(axis=1)
        distances = 1 - similarity
    elif query_features.ndim == 2 and target_features.ndim == 3:
        similarity = np.einsum("nd,mqd->nmq", query_features, target_features).max(axis=2)
        distances = 1 - similarity
    else:
        distances = cdist(query_features, target_features, metric="cosine")

    nearest_indices = np.argsort(distances, axis=1)[:, :k]
    nearest_distances = np.take_along_axis(distances, nearest_indices, axis=1)
    return nearest_indices, nearest_distances


def run_nearest_neighbor_analysis():
    dataset = COCODataset()

    models = {
        "CLIP": CLIPModel(),
        "BLIP": BLIPModel(),
        "BLIP2": BLIP2Model(),
    }

    results = {}

    for model_name, model in models.items():
        print(f"\nProcessing {model_name} nearest neighbor analysis...")

        data = extract_nn_embeddings(model, dataset, model_name, num_samples=NUM_SAMPLES_NEAREST_NEIGHBOR)
        image_features = data["image_features"]
        text_features = data["text_features"]
        captions = data["captions"]
        image_paths = data["image_paths"]
        image_ids = data["image_ids"]

        image_to_text_indices, image_to_text_distances = find_nearest_neighbors(image_features, text_features, k=NUM_NEIGHBORS)
        text_to_image_indices, text_to_image_distances = find_nearest_neighbors(text_features, image_features, k=NUM_NEIGHBORS)

        nn_results = {
            "num_samples": len(captions),
            "image_to_text": [],
            "text_to_image": [],
        }

        limit = min(NUM_QUALITATIVE_EXAMPLES, len(captions))
        for i in range(limit):
            nn_results["image_to_text"].append(
                {
                    "query_image_id": image_ids[i],
                    "query_image_path": image_paths[i],
                    "ground_truth_caption": captions[i],
                    "nearest_captions": [captions[j] for j in image_to_text_indices[i]],
                    "distances": image_to_text_distances[i].tolist(),
                }
            )

            nn_results["text_to_image"].append(
                {
                    "query_caption": captions[i],
                    "ground_truth_image_id": image_ids[i],
                    "nearest_images": [
                        {
                            "image_id": image_ids[j],
                            "image_path": image_paths[j],
                            "reference_caption": captions[j],
                        }
                        for j in text_to_image_indices[i]
                    ],
                    "distances": text_to_image_distances[i].tolist(),
                }
            )

        results[model_name] = nn_results
        print(f"Nearest neighbor analysis completed for {model_name}")

    with open(os.path.join(RESULTS_DIR, "nearest_neighbor_results.json"), "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\nNearest neighbor results saved to nearest_neighbor_results.json")


def analyze_compositional_generalization():
    test_cases = [
        ("red bus", "blue bus"),
        ("two dogs", "one dog"),
        ("man behind horse", "man riding horse"),
        ("cat on couch", "cat under couch"),
        ("black cat", "black dog"),
    ]

    dataset = COCODataset()

    models = {
        "CLIP": CLIPModel(),
        "BLIP": BLIPModel(),
        "BLIP2": BLIP2Model(),
    }

    pairs = dataset.get_image_caption_pairs(max_samples=NUM_SAMPLES_NEAREST_NEIGHBOR)
    results = {}

    for model_name, model in models.items():
        print(f"\nAnalyzing compositional generalization for {model_name}...")

        dataset_text_features = np.array([model.extract_text_features(pair["caption"]) for pair in pairs])
        model_results = []

        for text1, text2 in test_cases:
            feat1 = model.extract_text_features(text1)
            feat2 = model.extract_text_features(text2)

            similarity = 1 - cdist([feat1], [feat2], metric="cosine")[0][0]
            nn_1, _ = find_nearest_neighbors(np.array([feat1]), dataset_text_features, k=NUM_NEIGHBORS)
            nn_2, _ = find_nearest_neighbors(np.array([feat2]), dataset_text_features, k=NUM_NEIGHBORS)

            model_results.append(
                {
                    "pair": [text1, text2],
                    "similarity": float(similarity),
                    "nearest_dataset_captions_for_first": [pairs[idx]["caption"] for idx in nn_1[0]],
                    "nearest_dataset_captions_for_second": [pairs[idx]["caption"] for idx in nn_2[0]],
                }
            )

        results[model_name] = model_results

    with open(os.path.join(RESULTS_DIR, "compositional_analysis.json"), "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\nCompositional generalization analysis saved to compositional_analysis.json")

if __name__ == "__main__":
    run_nearest_neighbor_analysis()
    analyze_compositional_generalization()
