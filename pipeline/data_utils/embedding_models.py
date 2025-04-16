import logging
import torch

logger = logging.getLogger(__name__)

SUPPORTED_MODELS = {
    "clip": [
        "ViT-B/32",
        "ViT-B/16",
        "RN50",
        "ViT-L/14",
    ],
    "open_clip": [
        "ViT-B-32"
        "ViT-B-16",
        "ViT-L-14",
    ],
    "SD": [
        "15"
    ]
}
TEXT_AND_IMAGE_MODELS = ["clip", "open_clip"]
TEXT_MODELS = ["SD"]
SD_MAP = {
    "15": "stable-diffusion-v1-5/stable-diffusion-v1-5"
}
class EmbeddingExtractor:
    def __init__(self, model_name, model_dir) -> None:
        self.model_name = model_name
        self.model_dir = model_dir
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        self.model, self.preprocessor, self.tokenizer, self.token_max_length = self.load_model(model_name, self.device)
    
    @staticmethod
    def load_model(model_name, device, model_path=None):
        model_name = model_name.replace("~", "/")
        if model_name in SUPPORTED_MODELS['clip']:
            import clip
            model, preprocessor = clip.load(model_name, device=device, download_root=model_path)
            original_tokenizer = clip.tokenize
            tokenizer = lambda x: original_tokenizer(x, truncate=True)
            token_max_length = 77
        elif model_name in SUPPORTED_MODELS['open_clip']:
            import open_clip
            if model_name == "ViT-L-14":
                pretrained = "laion2b_s32b_b82k"
            else:
                pretrained = "laion2b_s34b_b79k"
            model, _, preprocessor = open_clip.create_model_and_transforms(model_name, device=device, cache_dir=model_path, pretrained=pretrained) ##FIXME maybe allow specifying pretrained? probs not though? I think the same. laion2B-39B-b160k",
            tokenizer = open_clip.get_tokenizer(model_name)
            token_max_length = tokenizer.context_length
        elif model_name in SUPPORTED_MODELS['SD']:
            from diffusers import StableDiffusionPipeline
            true_model_name = SD_MAP[model_name]
            model = StableDiffusionPipeline.from_pretrained(true_model_name, 
                                                            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                                                            cache_dir=model_path).to(device)
            tokenizer = lambda x: x
            preprocessor = lambda x: x
            token_max_length = 77
            model.encode_image = None
            model.encode_text = lambda x: model.encode_prompt(x, device=device, num_images_per_prompt=1, do_classifier_free_guidance=True)[0]
        else:
            raise RuntimeError(f"Model type {model_name} not supported.")
    
        try:
            model.eval()
        except Exception as e:
            logger.info(f"Failed move model to eval mode: {e}")
        return model, preprocessor, tokenizer, token_max_length

def embed_text(model, tokenizer, text, device):
    if not isinstance(text, torch.Tensor):
        text_embeddings = tokenizer(text if isinstance(text, list) else [text])
    else:
        text_embeddings = text
    
    try:
        text_embeddings = text_embeddings.to(device)
    except Exception:
        pass

    with torch.no_grad(), torch.cuda.amp.autocast():
        text_features = model.encode_text(text_embeddings)

    if isinstance(text_embeddings, torch.Tensor):
        text_embeddings = text_embeddings.detach().cpu()

    return text_features, text_embeddings

def embed_image(model, preprocess, img, device):
    if isinstance(img, list):
        if not isinstance(img[0], torch.Tensor):
            img = [preprocess(i).to(device) for i in img]
        img = torch.stack(img, dim=0)
    else:
        if not isinstance(img, torch.Tensor):
            img = preprocess(img)
        img = img.unsqueeze(0)

    try:
        img = img.to(device)
    except Exception:
        pass

    with torch.no_grad(), torch.cuda.amp.autocast():
        image_features = model.encode_image(img)

    return image_features, img.detach().cpu()