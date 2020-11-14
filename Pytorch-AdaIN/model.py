import torch
from torch import nn
import torch.nn.functional as F
import torchvision.models as models # Pretrained VGG-19
import copy
from adain import AdaIN, compute_mean_std

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vgg = models.vgg19(pretrained=True).features().to(device).eval()

def build_encoder(base_model=vgg):
    base_model = copy.deepcopy(base_model)
    i = 0
    model = nn.Sequential([])
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
        if name[:4] == "conv" and name[5] != "1":
            model.add_module("pad_{}".format(i), nn.ReflectionPad2d((1,1,1,1)))
        model.add_module(name, layer)
        if name == "relu_10": #ReLU_4_1
            break
    return model

encoder = build_encoder(base_model=vgg)
decoder = nn.Sequential(
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 256, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 128, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 64, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 3, (3, 3)))

class Net(nn.Module):

    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        enc_layers = list(encoder.children())
        self.enc_1 = nn.Sequential(*enc_layers[:4]) # relu_1_1
        self.enc_2 = nn.Sequential(*enc_layers[4:11]) # relu_2_1
        self.enc_3 = nn.Sequential(*enc_layers[11:18]) # relu_3_1
        self.enc_4 = nn.Sequential(*enc_layers[18:31]) # relu_4_1 (final)

        for param in self.encoder.parameters(): # Freeze VGG encoder
            param.requires_grad = False

        for name in ['enc_{}'.format(i) for i in range(1,5)]: # Freeze separate encoder layers
            for param in getattr(self, name).parameters():
                param.requires_grad = False

    def encode(self, input):
        return self.encoder(input)

    def encode_step(self, input):
        out = [input]
        for i in range(4):
            layer = getattr(self, 'enc_{}'.format(i+1))
            out.append(layer(out[-1]))
        return out[1:] # outputs enc_1, enc_2, enc_3, enc_4 output in a list / for style_loss calc

    def content_loss(self, input, target):
        assert (input.size()==target.size()), "Input and target size mismatch"
        assert (target.requires_grad = False), "Warning: freeze target gradients"
        return nn.MSELoss(input, target)

    def style_loss(self, input, target):
        assert (input.size()==target.size()), "Input and target size mismatch"
        assert (target.requires_grad = False), "Warning: freeze target gradients"
        input_mean, input_std = compute_mean_std(input)
        target_mean, target_std = compute_mean_std(target)
        return nn.MSELoss(input_mean, target_mean) + nn.MSELoss(input_std, target_std)

    def forward(self, content, style, alpha=1.0, output_image=False):
        assert 0 <= alpha <= 1, "Invalid interpolation factor"
        style_features = self.encode_step(style)
        content_feature = self.encode(content)

        t = AdaIN(content_feature, style_features[-1])
        t = alpha * t + (1-alpha) * content_feature

        g_t = self.decoder(t)

        if output_image: # View generated image
            return g_t

        g_t_features = self.encode_step(g_t) # Encode once again to compute loss

        loss_c = content_loss(g_t_features[-1], t)
        loss_s = style_loss(g_t_features[0], style_features[0])
        for i in range(1, 4):
            loss_c += style_loss(g_t_features[i], style_features[i])

        return loss_c, loss_s
