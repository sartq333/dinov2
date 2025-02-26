from dinov2.models.vision_transformer import vit_base
import torch
from PIL import Image
import requests
from torchvision import transforms

model = vit_base(img_size=518, patch_size=14, init_values=1.0, ffn_layer="mlp", block_chunks=0)

state_dict = torch.hub.load_state_dict_from_url("https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth",
                                                map_location="cpu",
                                                )

model.load_state_dict(state_dict)

url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open(requests.get(url, stream=True).raw)

transformations = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], # these are RGB mean+std values
            std=[0.229, 0.224, 0.225]  # across a large photo dataset.
        )
])

pixel_values = transformations(image).unsqueeze(0) # insert batch dimensions

outputs = model.forward_features(pixel_values)

print("Outputs: ", outputs)