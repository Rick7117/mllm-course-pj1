import json
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from tqdm import tqdm

from config import EMBEDDINGS_DIR, NUM_SAMPLES_VISUALIZATION, TSNE_RANDOM_STATE, VISUALIZATION_DIR
from dataset import COCODataset
from models import BLIP2Model, BLIPModel, CLIPModel


def pooled_features(features):
    if features.ndim == 3:
        return features.mean(axis=1)
    return features


def normalize_features(features):
    norms = np.linalg.norm(features, axis=-1, keepdims=True)
    return features / np.clip(norms, 1e-12, None)


def cosine_similarity_matrix(a, b):
    a = normalize_features(a)
    b = normalize_features(b)

    if a.ndim == 3 and b.ndim == 2:
        similarity = np.einsum("nqd,md->nqm", a, b)
        return similarity.max(axis=1)
    if a.ndim == 2 and b.ndim == 3:
        similarity = np.einsum("nd,mqd->nmq", a, b)
        return similarity.max(axis=2)
    if a.ndim == 3 and b.ndim == 3:
        similarity = np.einsum("nqd,mrd->nqmr", a, b)
        return similarity.max(axis=(1, 3))

    return a @ b.T


def extract_visualization_embeddings(model, dataset, model_name, num_samples=500):
    cache_name = getattr(model, "cache_name", model_name.lower())
    cache_path = os.path.join(EMBEDDINGS_DIR, f"{cache_name}_{num_samples}_viz_embeddings.pkl")

    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    pairs = dataset.get_image_caption_pairs(max_samples=num_samples)

    image_features = []
    text_features = []
    labels = []
    captions = []

    for pair in tqdm(pairs, desc=f"Extracting {model_name} viz features"):
        img_feat = model.extract_image_features(pair["image_path"])
        txt_feat = model.extract_text_features(pair["caption"])

        image_features.append(img_feat)
        text_features.append(txt_feat)
        labels.append(pair["image_id"])
        captions.append(pair["caption"])

    result = {
        "image_features": np.array(image_features),
        "text_features": np.array(text_features),
        "labels": labels,
        "captions": captions,
    }

    with open(cache_path, "wb") as f:
        pickle.dump(result, f)

    return result


def analyze_embeddings(image_features, text_features):
    similarity = cosine_similarity_matrix(image_features, text_features)
    pair_similarity = np.diag(similarity)
    nearest_text_index = np.argmax(similarity, axis=1)
    nearest_image_index = np.argmax(similarity, axis=0)

    pooled_image_features = pooled_features(image_features)
    pooled_text_features = pooled_features(text_features)

    image_centroid = pooled_image_features.mean(axis=0)
    text_centroid = pooled_text_features.mean(axis=0)
    centroid_distance = float(np.linalg.norm(image_centroid - text_centroid))

    shuffled_text_features = np.roll(text_features, shift=1, axis=0)
    shuffled_similarity_matrix = cosine_similarity_matrix(image_features, shuffled_text_features)
    shuffled_similarity = np.diag(shuffled_similarity_matrix)

    return {
        "avg_pair_similarity": float(pair_similarity.mean()),
        "median_pair_similarity": float(np.median(pair_similarity)),
        "avg_random_pair_similarity": float(shuffled_similarity.mean()),
        "image_to_text_top1_alignment": float(np.mean(nearest_text_index == np.arange(len(image_features)))),
        "text_to_image_top1_alignment": float(np.mean(nearest_image_index == np.arange(len(text_features)))),
        "centroid_distance": centroid_distance,
    }


def visualize_embeddings(model_name, image_features, text_features, labels, method="pca"):
    image_features = pooled_features(image_features)
    text_features = pooled_features(text_features)
    all_features = np.vstack([image_features, text_features])
    all_labels = np.array(["image"] * len(image_features) + ["text"] * len(text_features))

    if method == "pca":
        reducer = PCA(n_components=2)
    elif method == "tsne":
        perplexity = min(30, max(5, len(all_features) // 10))
        reducer = TSNE(n_components=2, random_state=TSNE_RANDOM_STATE, perplexity=perplexity)
    else:
        reducer = PCA(n_components=2)

    reduced = reducer.fit_transform(all_features)

    plt.figure(figsize=(12, 8))

    for i in range(len(labels)):
        img_point = reduced[i]
        txt_point = reduced[i + len(labels)]
        plt.plot(
            [img_point[0], txt_point[0]],
            [img_point[1], txt_point[1]],
            color="gray",
            linestyle="--",
            linewidth=0.5,
            alpha=0.25,
        )

    mask_image = all_labels == "image"
    mask_text = all_labels == "text"

    plt.scatter(reduced[mask_image, 0], reduced[mask_image, 1], label="Image", alpha=0.7, s=35, color="blue")
    plt.scatter(reduced[mask_text, 0], reduced[mask_text, 1], label="Text", alpha=0.7, s=35, color="red")

    plt.title(f"{model_name} Embedding Visualization ({method.upper()})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALIZATION_DIR, f"{model_name}_{method}_viz.png"), dpi=300, bbox_inches="tight")
    plt.close()


def run_visualization():
    dataset = COCODataset()

    models = {
        "CLIP": CLIPModel(),
        "BLIP": BLIPModel(),
        "BLIP2": BLIP2Model(),
    }

    analysis_results = {}

    for model_name, model in models.items():
        print(f"\nProcessing {model_name} visualization...")

        result = extract_visualization_embeddings(model, dataset, model_name, num_samples=NUM_SAMPLES_VISUALIZATION)
        image_features = result["image_features"]
        text_features = result["text_features"]
        labels = result["labels"]

        print(f"Image features shape: {image_features.shape}")
        print(f"Text features shape: {text_features.shape}")

        visualize_embeddings(model_name, image_features, text_features, labels, method="pca")
        visualize_embeddings(model_name, image_features, text_features, labels, method="tsne")
        analysis_results[model_name] = analyze_embeddings(image_features, text_features)

        print(f"Visualizations saved for {model_name}")

    with open(os.path.join(VISUALIZATION_DIR, "representation_analysis.json"), "w") as f:
        json.dump(analysis_results, f, indent=2, ensure_ascii=False)

    print("\nRepresentation analysis saved to representation_analysis.json")

if __name__ == "__main__":
    run_visualization()
