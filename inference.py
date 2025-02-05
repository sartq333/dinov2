from dinov2.models.vision_transformer import vit_base
import torch
from PIL import Image
import requests
from torchvision import transforms

model = vit_base()

state_dict = torch.hub.load_state_dict_from_url("https://dl.fbaipublicfiles.com/dinvov2/dinov2_vitb14_pretrain.pth",
                                                map_location="cpu",
                                                headers={"User-Agent": "Mozilla/5.0"})

model.load_state_dict(state_dict)

url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open(requests.get(url, stream=True).raw)

transformations = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
])

pixel_values = transformations(image).unsqueeze(0) # insert batch dimensions

outputs = model.forward_features(pixel_values)

print("Outputs: ", outputs)