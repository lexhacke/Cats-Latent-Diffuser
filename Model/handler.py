from diffusers import AutoencoderKL
from transformers import CLIPProcessor, CLIPModel
from model import Model
from noise_scheduler import NoiseSchedule
import torch
import base64
from typing import Any, Dict

LDM = True
image_size = 512
latent_size = 64
filters = [64, 128, 256, 512]
latent_dim = 4
t_dim = 512
T = 1000
depth = 2

class CLIP:
    def __init__(self):
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.model.eval()
        for name, param in self.model.named_parameters():
            param.requires_grad = False

    @torch.inference_mode()
    def embed_images(self, images):
        image = self.processor(images=images, return_tensors="pt").to(self.model.device)
        return self.model.get_image_features(**image)
    
    @torch.inference_mode()
    def embed_text(self, text):
        text = self.processor(text, padding=True, return_tensors="pt").to(self.model.device)
        return self.model.get_text_features(**text)

class Inference:
    def __init__(self):
        self.clip = CLIP()
        self.ae = AutoencoderKL.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="vae").to('cuda' if torch.cuda.is_available() else "cpu")
        self.ae.eval()
        for name, param in self.ae.named_parameters():
            param.requires_grad = False
        self.unet = Model(T=T, filters=[64,128,256,512], t_dim=t_dim, depth=depth, LDM=LDM)
        self.unet.load_state_dict(torch.load("unet.pt", weights_only=False, map_location=torch.device('cpu')))
        self.unet.eval()
        for name, param in self.unet.named_parameters():
            param.requires_grad = False
        self.noise_scheduler = NoiseSchedule(T=1000, shape=(4,64,64), ddim_mod=50, trainer_mode=True)
        self.target_vector = self.clip.embed_text("A photo of a cat")[0] 
        self.target_vector = self.target_vector / self.target_vector.norm(p=2, dim=-1, keepdim=True)
    @torch.inference_mode()
    def __call__(self, num_images=8):
        imgs = self.noise_scheduler.generate(self.unet, num_images=num_images, device='cpu')
        max_img = None
        max_score = -1
        images = []
        for img in imgs:
            image = self.ae.decode(img.unsqueeze(0) / self.ae.config.scaling_factor)[0][0].cpu().permute(1,2,0)/2 + 0.5
            image = torch.clamp(image, 0.0, 1.0)
            images.append(image)
        embeddings = self.clip.embed_images(images)
        scores = (embeddings / embeddings.norm(p=2, dim=-1, keepdim=True)) @ self.target_vector.T
        i = torch.argmax(scores).item()
        return images[i], scores[i], scores  

class EndpointHandler:
    def __init__(self, path: str = ""):
        # path -> repo directory on the endpoint container
        # you can read files via Path(path)/"unet.pt" if needed
        self.engine = Inference()

    def __call__(self) -> Dict[str, Any]:
        png_bytes, score = self.engine(num_images=1)
        b64 = base64.b64encode(png_bytes).decode("utf-8")
        return {"image": b64, "score": float(score)}
