import json
import os

from tqdm import tqdm

from config import NUM_QUALITATIVE_EXAMPLES, NUM_SAMPLES_CAPTIONING, RESULTS_DIR
from dataset import COCODataset
from evaluator import CaptionEvaluator
from models import BLIP2Model, BLIPModel


def run_captioning():
    dataset = COCODataset()
    evaluator = CaptionEvaluator()

    models = {
        "BLIP": BLIPModel(),
        "BLIP2": BLIP2Model(),
    }

    image_ids = dataset.get_all_image_ids()[:NUM_SAMPLES_CAPTIONING]
    output_path = os.path.join(RESULTS_DIR, "captioning_results.json")
    results = {}

    if os.path.exists(output_path):
        with open(output_path, "r") as f:
            results = json.load(f)

    for model_name, model in models.items():
        if model_name in results:
            print(f"\nSkipping {model_name}, existing results found.")
            continue

        print(f"\nProcessing {model_name}...")

        predictions = {}
        ground_truths = {}
        qualitative_examples = []

        for image_id in tqdm(image_ids, desc=f"Generating captions with {model_name}"):
            image_path = dataset.get_image_path(image_id)
            captions = dataset.get_captions(image_id)

            if not image_path or not captions:
                continue

            generated_caption = model.generate_caption(image_path)
            predictions[str(image_id)] = [generated_caption]
            ground_truths[str(image_id)] = captions

            if len(qualitative_examples) < NUM_QUALITATIVE_EXAMPLES:
                qualitative_examples.append(
                    {
                        "image_id": image_id,
                        "image_path": image_path,
                        "prediction": generated_caption,
                        "ground_truths": captions,
                    }
                )

        metrics = evaluator.evaluate(predictions, ground_truths)
        results[model_name] = {
            "num_samples": len(predictions),
            "metrics": metrics,
            "qualitative_examples": qualitative_examples,
        }

        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"Results for {model_name}:")
        for metric, score in metrics.items():
            if isinstance(score, dict):
                print(f"  {metric}: {score}")
            else:
                print(f"  {metric}: {score:.4f}")

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\nResults saved to captioning_results.json")

if __name__ == "__main__":
    run_captioning()
