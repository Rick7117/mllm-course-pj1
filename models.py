import torch
import open_clip
from transformers import BlipProcessor, BlipForImageTextRetrieval, Blip2Processor, Blip2ForConditionalGeneration, BlipForConditionalGeneration
from PIL import Image


class CLIPModel:
    def __init__(self, model_name='ViT-L-14', checkpoint='laion2b_s32b_b82k'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_name, pretrained=checkpoint)
        self.model.to(self.device)
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer(model_name)
    
    def extract_image_features(self, image_path):
        image = self.preprocess(Image.open(image_path).convert('RGB')).unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = self.model.encode_image(image)
            features /= features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy().flatten()
    
    def extract_text_features(self, text):
        tokens = self.tokenizer([text]).to(self.device)
        with torch.no_grad():
            features = self.model.encode_text(tokens)
            features /= features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy().flatten()


class BLIPModel:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.model = BlipForImageTextRetrieval.from_pretrained("Salesforce/blip-itm-base-coco")
        self.model.to(self.device)
        self.model.eval()
        self.caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        self.caption_model.to(self.device)
        self.caption_model.eval()
    
    def extract_image_features(self, image_path):
        image = Image.open(image_path).convert('RGB')
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.vision_model(**inputs)
            image_embeds = outputs.last_hidden_state[:, 0, :]
            image_features = self.model.vision_proj(image_embeds)
            image_features /= image_features.norm(dim=-1, keepdim=True)
        return image_features.cpu().numpy().flatten()
    
    def extract_text_features(self, text):
        inputs = self.processor(text=[text], return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.text_encoder(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask)
            text_embeds = outputs.last_hidden_state[:, 0, :]
            text_features = self.model.text_proj(text_embeds)
            text_features /= text_features.norm(dim=-1, keepdim=True)
        return text_features.cpu().numpy().flatten()
    
    def generate_caption(self, image_path, max_length=50):
        image = Image.open(image_path).convert('RGB')
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.caption_model.generate(**inputs, max_length=max_length)
        return self.processor.decode(outputs[0], skip_special_tokens=True)


class BLIP2Model:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            "Salesforce/blip2-opt-2.7b",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        self.model.to(self.device)
        self.model.eval()
    
    def extract_image_features(self, image_path):
        image = Image.open(image_path).convert('RGB')
        inputs = self.processor(images=image, return_tensors="pt").to(self.device, dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
        with torch.no_grad():
            image_embeds = self.model.vision_model(**inputs).last_hidden_state
            batch_size = image_embeds.shape[0]
            query_tokens = self.model.query_tokens.expand(batch_size, -1, -1)
            query_output = self.model.qformer(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=torch.ones(image_embeds.shape[:-1], device=self.device, dtype=torch.long)
            )
            projected_queries = self.model.language_projection(query_output.last_hidden_state)
            image_features = projected_queries.mean(dim=1)
            image_features /= image_features.norm(dim=-1, keepdim=True)
        return image_features.cpu().numpy().flatten()
    
    def extract_text_features(self, text):
        inputs = self.processor(text=[text], return_tensors="pt").to(self.device)
        with torch.no_grad():
            text_embeds = self.model.language_model.get_input_embeddings()(inputs.input_ids)
            text_embeds = text_embeds.to(self.device)
            attention_mask = inputs.attention_mask.unsqueeze(-1).to(text_embeds.dtype)
            text_features = (text_embeds * attention_mask).sum(dim=1) / attention_mask.sum(dim=1).clamp(min=1.0)
            text_features /= text_features.norm(dim=-1, keepdim=True)
        return text_features.cpu().numpy().flatten()
    
    def generate_caption(self, image_path, max_length=50):
        image = Image.open(image_path).convert('RGB')
        inputs = self.processor(images=image, return_tensors="pt").to(self.device, dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_length=max_length)
        return self.processor.decode(outputs[0], skip_special_tokens=True)
