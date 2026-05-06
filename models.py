import os

import torch
from PIL import Image
from transformers import (
    AddedToken,
    Blip2ForConditionalGeneration,
    Blip2ForImageTextRetrieval,
    Blip2Processor,
    BlipForConditionalGeneration,
    BlipForImageTextRetrieval,
    BlipProcessor,
    CLIPModel as HFCLIPModel,
    CLIPProcessor,
)

from config import (
    BLIP2_CAPTION_MODEL_DIR,
    BLIP2_RETRIEVAL_MODEL_DIR,
    BLIP_CAPTION_MODEL_DIR,
    BLIP_RETRIEVAL_MODEL_DIR,
    CLIP_MODEL_DIR,
)


def _resolve_model_path(local_path, fallback_repo_id):
    return local_path if os.path.isdir(local_path) and os.listdir(local_path) else fallback_repo_id


def _move_inputs(inputs, device, float_dtype=None):
    moved = {}
    for key, value in inputs.items():
        if hasattr(value, "to"):
            if float_dtype is not None and torch.is_floating_point(value):
                moved[key] = value.to(device=device, dtype=float_dtype)
            else:
                moved[key] = value.to(device=device)
        else:
            moved[key] = value
    return moved


class CLIPModel:
    def __init__(self):
        self.cache_name = "clip_vit_base_patch32_modelscope"
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_path = _resolve_model_path(CLIP_MODEL_DIR, "openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained(model_path)
        self.model = HFCLIPModel.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
    
    def extract_image_features(self, image_path):
        image = Image.open(image_path).convert('RGB')
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = _move_inputs(inputs, self.device)
        with torch.no_grad():
            vision_outputs = self.model.vision_model(
                pixel_values=inputs["pixel_values"],
                return_dict=True,
            )
            features = self.model.visual_projection(vision_outputs.pooler_output)
            features /= features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy().flatten()
    
    def extract_text_features(self, text):
        inputs = self.processor(text=[text], return_tensors="pt", padding=True)
        inputs = _move_inputs(inputs, self.device)
        with torch.no_grad():
            text_outputs = self.model.text_model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                return_dict=True,
            )
            features = self.model.text_projection(text_outputs.pooler_output)
            features /= features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy().flatten()


class BLIPModel:
    def __init__(self):
        self.cache_name = "blip_itm_base_coco_modelscope"
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        retrieval_path = _resolve_model_path(BLIP_RETRIEVAL_MODEL_DIR, "Salesforce/blip-itm-base-coco")
        caption_path = _resolve_model_path(BLIP_CAPTION_MODEL_DIR, "Salesforce/blip-image-captioning-base")

        self.retrieval_processor = BlipProcessor.from_pretrained(retrieval_path)
        self.model = BlipForImageTextRetrieval.from_pretrained(retrieval_path)
        self.caption_processor = BlipProcessor.from_pretrained(caption_path)
        self.caption_model = BlipForConditionalGeneration.from_pretrained(caption_path)

        self.model.to(self.device)
        self.caption_model.to(self.device)
        self.model.eval()
        self.caption_model.eval()
    
    def extract_image_features(self, image_path):
        image = Image.open(image_path).convert('RGB')
        inputs = self.retrieval_processor(images=image, return_tensors="pt")
        inputs = _move_inputs(inputs, self.device)
        with torch.no_grad():
            outputs = self.model.vision_model(**inputs)
            image_embeds = outputs.last_hidden_state[:, 0, :]
            image_features = self.model.vision_proj(image_embeds)
            image_features /= image_features.norm(dim=-1, keepdim=True)
        return image_features.cpu().numpy().flatten()
    
    def extract_text_features(self, text):
        inputs = self.retrieval_processor(text=[text], return_tensors="pt")
        inputs = _move_inputs(inputs, self.device)
        with torch.no_grad():
            outputs = self.model.text_encoder(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            )
            text_embeds = outputs.last_hidden_state[:, 0, :]
            text_features = self.model.text_proj(text_embeds)
            text_features /= text_features.norm(dim=-1, keepdim=True)
        return text_features.cpu().numpy().flatten()
    
    def generate_caption(self, image_path, max_length=50):
        image = Image.open(image_path).convert('RGB')
        inputs = self.caption_processor(images=image, return_tensors="pt")
        inputs = _move_inputs(inputs, self.device)
        with torch.no_grad():
            outputs = self.caption_model.generate(**inputs, max_length=max_length)
        return self.caption_processor.decode(outputs[0], skip_special_tokens=True).strip()


class BLIP2Model:
    def __init__(self):
        self.cache_name = "blip2_coco_modelscope"
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.float_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        retrieval_path = _resolve_model_path(BLIP2_RETRIEVAL_MODEL_DIR, "Salesforce/blip2-itm-vit-g")
        caption_path = _resolve_model_path(BLIP2_CAPTION_MODEL_DIR, "Salesforce/blip2-opt-2.7b")

        self.retrieval_processor = Blip2Processor.from_pretrained(retrieval_path)
        self.model = Blip2ForImageTextRetrieval.from_pretrained(
            retrieval_path,
            torch_dtype=self.float_dtype,
            low_cpu_mem_usage=True,
        )
        self.caption_processor = Blip2Processor.from_pretrained(caption_path)
        self.caption_model = Blip2ForConditionalGeneration.from_pretrained(
            caption_path,
            torch_dtype=self.float_dtype,
            low_cpu_mem_usage=True,
        )
        self._prepare_blip2_caption_processor()
        self.model.to(self.device)
        self.caption_model.to(self.device)
        self.model.eval()
        self.caption_model.eval()

    def _prepare_blip2_caption_processor(self):
        num_query_tokens = getattr(self.caption_model.config, "num_query_tokens", None)
        if num_query_tokens is not None:
            self.caption_processor.num_query_tokens = num_query_tokens

        if getattr(self.caption_model.config, "image_token_id", None) is None:
            tokenizer = self.caption_processor.tokenizer
            image_token = "<image>"
            if image_token not in tokenizer.get_vocab():
                tokenizer.add_tokens([AddedToken(image_token, special=True, normalized=False)])
                self.caption_model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=64)

            image_token_id = tokenizer.convert_tokens_to_ids(image_token)
            self.caption_model.config.image_token_id = image_token_id
            self.caption_model.generation_config.image_token_id = image_token_id
    
    def extract_image_features(self, image_path):
        image = Image.open(image_path).convert('RGB')
        inputs = self.retrieval_processor(images=image, return_tensors="pt")
        inputs = _move_inputs(inputs, self.device, self.float_dtype)
        with torch.no_grad():
            vision_outputs = self.model.vision_model(pixel_values=inputs["pixel_values"], return_dict=True)
            image_embeds = vision_outputs.last_hidden_state
            batch_size = image_embeds.shape[0]
            image_attention_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=self.device)
            query_tokens = self.model.query_tokens.expand(batch_size, -1, -1)
            query_output = self.model.qformer(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_attention_mask,
                return_dict=True,
            )
            image_features = self.model.vision_projection(query_output.last_hidden_state)
            image_features /= image_features.norm(dim=-1, keepdim=True)
        return image_features[0].cpu().numpy()
    
    def extract_text_features(self, text):
        inputs = self.retrieval_processor(text=[text], return_tensors="pt")
        inputs = _move_inputs(inputs, self.device)
        with torch.no_grad():
            query_embeds = self.model.embeddings(input_ids=inputs["input_ids"])
            text_outputs = self.model.qformer(
                query_embeds=query_embeds,
                query_length=0,
                attention_mask=inputs["attention_mask"],
                return_dict=True,
            )
            text_embeds = text_outputs.last_hidden_state[:, 0, :].to(dtype=self.model.text_projection.weight.dtype)
            text_features = self.model.text_projection(text_embeds)
            text_features /= text_features.norm(dim=-1, keepdim=True)
        return text_features.cpu().numpy().flatten()
    
    def generate_caption(self, image_path, max_length=50):
        image = Image.open(image_path).convert('RGB')
        prompt = "a photo of"
        inputs = self.caption_processor(images=image, text=prompt, return_tensors="pt")
        inputs = _move_inputs(inputs, self.device, self.float_dtype)
        with torch.no_grad():
            outputs = self.caption_model.generate(**inputs, max_new_tokens=max_length)
        caption = self.caption_processor.decode(outputs[0], skip_special_tokens=True).strip()
        if caption.lower().startswith(prompt):
            caption = caption[len(prompt):].strip(" ,.")
        return caption
