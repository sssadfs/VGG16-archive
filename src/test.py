from dataload import *
from model import *

image_path = "../tiny-imagenet-200/test/images/"
image_name = "test_1.JPEG"
image_dir = os.path.join(image_path, image_name)

image = Image.open(image_dir)

image = transformation(image)
image = image.unsqueeze(0)

mynet = VGG16(num_classes = 200)
mynet.load_state_dict(torch.load("../checkpoints/VGG16_epoch_1.pth",weights_only=False))

mynet.eval()
with torch.no_grad():
    output = mynet(image)

print(output.argmax(dim=1).item())