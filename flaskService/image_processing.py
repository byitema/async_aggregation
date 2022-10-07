from torchvision.transforms import transforms
import torch
import torchvision
from PIL import Image


def process_image(image, model):
    IMAGE_PREPROCESS = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    MODEL_ARCHITECTURE = "resnet18", "resnet34", "resnet50", "resnet101", "resnet152"

    torch.hub._validate_not_a_forked_repo = lambda a, b, c: True
    MODEL: torchvision.models.resnet.ResNet = torch.hub.load('pytorch/vision:v0.6.0', MODEL_ARCHITECTURE[model], pretrained=True)

    image = Image.open(image).convert('RGB')
    input_tensor = IMAGE_PREPROCESS(image)
    input_batch = input_tensor.unsqueeze(0)

    with torch.no_grad():
        output = MODEL(input_batch)

    result = torch.nn.functional.softmax(output[0], dim=0).tolist()

    return result
