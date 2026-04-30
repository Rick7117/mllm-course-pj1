import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
COCO_IMAGES_DIR = os.path.join(DATA_DIR, 'val2017')
COCO_ANNOTATIONS_DIR = os.path.join(DATA_DIR, 'annotations')

OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'output')
EMBEDDINGS_DIR = os.path.join(OUTPUT_DIR, 'embeddings')
RESULTS_DIR = os.path.join(OUTPUT_DIR, 'results')
VISUALIZATION_DIR = os.path.join(OUTPUT_DIR, 'visualizations')

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(COCO_IMAGES_DIR, exist_ok=True)
os.makedirs(COCO_ANNOTATIONS_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(VISUALIZATION_DIR, exist_ok=True)

NUM_SAMPLES_VISUALIZATION = 500
NUM_NEIGHBORS = 5
NUM_SAMPLES_CAPTIONING = 30
NUM_SAMPLES_NEAREST_NEIGHBOR = 500
NUM_QUALITATIVE_EXAMPLES = 20
TSNE_RANDOM_STATE = 42
