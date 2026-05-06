import numpy as np
import shutil
from scipy.spatial.distance import cdist


class RetrievalEvaluator:
    @staticmethod
    def compute_similarity_matrix(image_features, text_features):
        if image_features.ndim == 3:
            similarity = np.einsum("iqd,td->iqt", image_features, text_features)
            return similarity.max(axis=1)
        return 1 - cdist(image_features, text_features, metric='cosine')

    @staticmethod
    def compute_recall_at_k(image_features, text_features, image_to_text_mapping, k_list=[1, 5, 10]):
        similarity_matrix = RetrievalEvaluator.compute_similarity_matrix(image_features, text_features)
        
        image_to_text_recall = {k: 0 for k in k_list}
        text_to_image_recall = {k: 0 for k in k_list}
        
        num_images = len(image_features)
        
        for i in range(num_images):
            text_indices = np.argsort(similarity_matrix[i])[::-1]
            
            for k in k_list:
                top_k_texts = text_indices[:k]
                if image_to_text_mapping[i] in top_k_texts:
                    image_to_text_recall[k] += 1
        
        for j in range(len(text_features)):
            image_indices = np.argsort(similarity_matrix[:, j])[::-1]
            
            for k in k_list:
                top_k_images = image_indices[:k]
                if j in [image_to_text_mapping[i] for i in top_k_images]:
                    text_to_image_recall[k] += 1
        
        for k in k_list:
            image_to_text_recall[k] /= num_images
            text_to_image_recall[k] /= len(text_features)
        
        return image_to_text_recall, text_to_image_recall


class CaptionEvaluator:
    def __init__(self):
        from pycocoevalcap.bleu.bleu import Bleu
        from pycocoevalcap.cider.cider import Cider
        _ = shutil.which("java")

        self.bleu = Bleu(4)
        self.cider = Cider()

        try:
            from pycocoevalcap.rouge.rouge import Rouge
            self.rouge = Rouge()
        except Exception:
            self.rouge = None
    
    def evaluate(self, predictions, ground_truths):
        results = {}
        
        bleu_scores, _ = self.bleu.compute_score(ground_truths, predictions)
        results['BLEU-1'] = bleu_scores[0]
        results['BLEU-2'] = bleu_scores[1]
        results['BLEU-3'] = bleu_scores[2]
        results['BLEU-4'] = bleu_scores[3]
        
        cider_score, _ = self.cider.compute_score(ground_truths, predictions)
        results['CIDEr'] = cider_score

        if self.rouge is not None:
            try:
                rouge_score, _ = self.rouge.compute_score(ground_truths, predictions)
                results['ROUGE-L'] = rouge_score
            except Exception:
                pass
        
        return results
