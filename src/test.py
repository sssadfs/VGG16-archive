import os
from PIL import Image
from dataload import *

image_path = "../tiny-imagenet-200/test/images/"
image_name = "test_6.JPEG"
image_dir = os.path.join(image_path, image_name)

image = Image.open(image_dir)

image = transformation(image)
image = image.unsqueeze(0)

mynet = torch.load("../models/vgg16_pretrained.pth", weights_only=False)
print(mynet)
mynet.eval()
with torch.no_grad():
    output = mynet(image)

print(output.argmax(dim=1).item())