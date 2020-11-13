import torch
from torch import nn, optim
import torch.nn.functional as F

from PIL import Image
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
import torchvision.models as models # pretrained VGG19

import copy
# copy.copy(): Shallow Copy - List 내의 Mutable Objects들은 reference가 변하면 변한다. 겉은 다르지만 내용물은 같은 객체 생성.
# copy.deepcopy(): Deep Copy - List 내의 Mutable Objects들도 다른 reference가 변할 때 불변. 내용물까지 새로운 객체를 생성.

from loss import StyleLoss, ContentLoss
from utils import Normalization

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

imsize = 512 if torch.cuda.is_available() else 128 # Flexible image resolution depending on GPU

loader = transforms.Compose([
    transforms.Resize(imsize), # Default: Bilinear Interpolation
    transforms.ToTensor() # PIL to Tensor
])

def image_loader(image_name):
    image = Image.open(image_name) # PIL
    # Dummy dimension required to fit CNN
    image = loader(image).unsqueeze(0) # transform and make dummy dimension (Add dimension at given position 0)
    return image.to(device, torch.float)

style_img = image_loader("img/VanGoghCafe.jpg")
content_img = image_loader("img/702656.png")

###############################################

unloader = transforms.ToPILImage()

def imshow(tensor, title=None):
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)

###############################################

base_model = models.vgg19(pretrained=True).features.to(device).eval() # Freeze gradient
base_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
base_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

content_layers = ['conv_4']
style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

def get_style_model_and_loss(base_model, base_mean, base_std, style_img, content_img, content_layers=content_layers, style_layers=style_layers):

    base_model = copy.deepcopy(base_model) # Separate Object that doesn't affect each other
    norm = Normalization(base_mean, base_std).to(device)

    content_losses, style_losses = [], []

    model = nn.Sequential(norm) # First Layer = Normalization Layer

    i = 0 # Count CNN layers
    for layer in base_model.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = "conv_{}".format(i)
        elif isinstance(layer, nn.ReLU):
            name = "relu_{}".format(i)
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = "pool_{}".format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = "bn_{}".format(i)
        else:
            raise RuntimeError("Unrecognized layer: {}".format(layer.__class__.__name__))

        model.add_module(name, layer) # Sequentially stack layers of VGG to our new model (Copy most of them, and insert Losses in the right place)

        if name in content_layers:
            target = model(content_img).detach() # Feature map of content img so far
            content_loss = ContentLoss(target) # input is directly fed
            model.add_module("content_loss_{}".format(i), content_loss) # Add a layer that computes loss and returns the original input (Like identity operation in a sense)
            content_losses.append(content_loss)

        if name in style_layers:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module("style_loss_{}".format(i), style_loss) # Again, compute the gradient and returns input as is
            style_losses.append(style_loss)

        # Get rid of unnecessary layers after style and content loss
        for i in range(len(model)-1, -1, -1):
            if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
                break

        model = model[:(i+1)]

        return model, style_losses, content_losses

def input_optimizer(input_img):
    optimizer = optim.LBFGS([input_img.requires_grad_()]) # We optimize not the parameters of the model but the input content image itself
    return optimizer

def run_style_transfer(base_model, base_mean, base_std, content_img, style_img, input_img, num_steps=300, style_weight=1e6, content_weight=1):
    model, style_losses, content_losses = get_style_model_and_loss(base_model, base_mean, base_std, style_img, content_img)
    optimizer = input_optimizer(input_img)

    run = [0]

    while run[0] <= num_steps:

        def closure():
            input_img.data.clamp_(0,1)
            optimizer.zero_grad()

            model(input_img) # Style and Content Losses are all computed by Autograd

            style_score=0
            content_score=0

            for sl in style_losses:
                style_score += sl.loss # get self.loss
            for cl in content_losses:
                content_score += cl.loss # get self.loss

            loss = style_weight * style_score + content_weight * content_score
            loss.backward()

            run[0] += 1
            if run[0] % 50 == 0:
                print("Current run {}".format(run))
                print("Style Loss: {:.4f} / Content Loss: {:.4f}".format(style_score.item(), content_score.item()))
                print()

            return style_score, content_score

        optimizer.step(closure)

    input_img.data.clamp_(0,1) # Finally make sure it's between 0 and 1 for visualization purpose
    return input_img

input_img = content_img.clone()

output = run_style_transfer(base_model, base_mean, base_std, content_img, style_img, input_img)

plt.figure()
imshow(output, title='Output Image')
plt.ioff()
plt.show()
