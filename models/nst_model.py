# -*- coding: utf-8 -*-
"""

nst_model

"""

from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as tt
import torch.nn.functional as F
import torch
import torch.nn as nn

from PIL import Image
from io import BytesIO

class NST_Dataset(Dataset):
  def __init__(self, content_dir, style_dir):
    super().__init__()
    self.transform = tt.Compose([
                                   tt.Resize(512),
                                   tt.ToTensor(),
                                   tt.Normalize(mean=[0.485, 0.456, 0.406],
                                                std=[0.229, 0.224, 0.225])]
                                  )
    
    self.content_image = content_dir
    self.style_image = style_dir
    
  def __len__(self):
    return 1

  def __getitem__(self, idx):
    content_image = Image.open(self.content_image)
    style_image = Image.open(self.style_image)
    content_image = self.transform(content_image)
    style_image = self.transform(style_image)
    return content_image, style_image

def denorm(tensor):
    std = torch.Tensor([0.229, 0.224, 0.225]).reshape(-1, 1, 1)
    mean = torch.Tensor([0.485, 0.456, 0.406]).reshape(-1, 1, 1)
    res = torch.clamp(tensor * std + mean, 0, 1)
    return res

def mean_std(features):

  batch, channels = features.size()[:2]
  mean = features.reshape(batch, channels, -1).mean(dim=2).reshape(batch, channels, 1, 1)
  std = features.reshape(batch, channels, -1).std(dim=2).reshape(batch, channels, 1, 1) + 1e-6

  return mean, std

class AdaIN(nn.Module):
  def __init__(self):
    super().__init__()

  def mean_std(self, features):
     batch, channels = features.size()[:2]
     mean = features.reshape(batch, channels, -1).mean(dim=2).reshape(batch, channels, 1, 1)
     std = features.reshape(batch, channels, -1).std(dim=2).reshape(batch, channels, 1, 1) + 1e-6

     return mean, std

  def gauss(self, x):

    noise = 0.02 * torch.randn(x.size(), device=device)
    x = x + noise
    return x

  def forward(self, content, style, train=True):

    mean_content, std_content = self.mean_std(content)
    mean_style, std_style = self.mean_std(style)
    output = std_style * (content - mean_content) / std_content + mean_style
    if train:
      output = self.gauss(output)
    
    return output

class Encoder(nn.Module):
  def __init__(self):
    super().__init__()
    vgg = torchvision.models.vgg19(pretrained=False).features
    self.layer1 = vgg[:2]
    self.layer2 = vgg[2:7]
    self.layer3 = vgg[7:12]
    self.layer4 = vgg[12:21]
    for param in self.parameters():
      param.requires_grad = False
    
  def forward(self, input, output_last_feature=False):
    e1 = self.layer1(input)
    e2 = self.layer2(e1)
    e3 = self.layer3(e2)
    e4 = self.layer4(e3)
    if output_last_feature:
      return e4
    else:
      return e1, e2, e3, e4

class ReflectConv(nn.Module):
  def __init__(self, in_size, out_size, kernel_size=3, pad_size=1, activated=True):
    super().__init__()
    self.pad = nn.ReflectionPad2d((pad_size, pad_size, pad_size, pad_size))
    self.conv = nn.Conv2d(in_size, out_size, kernel_size)
    self.activated = activated

  def forward(self, input):
    out = self.pad(input)
    out = self.conv(out)
    if self.activated:
      return F.relu(out, True)
    else:
      return out

class Decoder(nn.Module):
  def __init__(self):
    super().__init__()
    self.layer1 = ReflectConv(512, 256, 3, 1)
    self.layer2 = ReflectConv(256, 256, 3, 1)
    self.layer3 = ReflectConv(256, 256, 3, 1)
    self.layer4 = ReflectConv(256, 256, 3, 1)
    self.layer5 = ReflectConv(256, 128, 3, 1)
    self.layer6 = ReflectConv(128, 128, 3, 1)
    self.layer7 = ReflectConv(128, 64, 3, 1)
    self.layer8 = ReflectConv(64, 64, 3, 1)
    self.layer9 = ReflectConv(64, 3, 3, 1, False)

  def forward(self, input):
    out = self.layer1(input)
    out = F.interpolate(out, scale_factor=2)
    out = self.layer2(out)
    out = self.layer3(out)
    out = self.layer4(out)
    out = self.layer5(out)
    out = F.interpolate(out, scale_factor=2)
    out = self.layer6(out)
    out = self.layer7(out)
    out = F.interpolate(out, scale_factor=2)
    out = self.layer8(out)
    out = self.layer9(out)
    return out

class Model(nn.Module):
  def __init__(self):
    super().__init__()
    self.encoder = Encoder() 
    self.decoder = Decoder()
    self.adain = AdaIN()

  def sample(self, content, style, alpha = 1.0):
    content_map = self.encoder(content, output_last_feature=True)
    style_map = self.encoder(style, output_last_feature=True)
    t = self.adain(content_map, style_map, False)
    t = alpha * t + (1 - alpha) * content_map 
    out = denorm(self.decoder(t))
    return out

  @staticmethod
  def ContentLoss(target, t):
    return F.mse_loss(target, t)

  @staticmethod
  def StyleLoss(target, output):
    Ls = 0
    for c, s in zip(target, output):
      c_mean, c_std = mean_std(c)
      s_mean, s_std = mean_std(s)
      Ls += F.l1_loss(c_mean, s_mean) + F.l1_loss(c_std, s_std)
    return Ls
    

  def forward(self, content, style, alpha = 1.0):
    content_encod = self.encoder(content, output_last_feature=True)
    style_encod = self.encoder(style, output_last_feature=True)
    t = self.adain(content_encod, style_encod)
    t = alpha * t + (1 - alpha) * content_encod 
    decod = self.decoder(t)

    content_features = self.encoder(decod, output_last_feature=True)
    content_encods = self.encoder(decod, output_last_feature=False)
    style_encods = self.encoder(style, output_last_feature=False)
    
    return self.ContentLoss(content_features, t), self.StyleLoss(content_encods, style_encods)

def tensortophoto(tensor):
  photo = tensor.squeeze(0).detach().cpu()
  unloader = tt.ToPILImage()
  photo = unloader(photo)

  # transform PIL image to send to telegram
  bio = BytesIO()
  bio.name = 'result.jpeg'
  photo.save(bio, 'JPEG')
  bio.seek(0)

  return bio

def get_transfer(content, style, percent, path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = tt.Compose([
                            tt.Resize(512),
                            tt.ToTensor(),
                            tt.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])]
                           )
    model = Model().to(device)
    state_dict = torch.load(path, map_location=device)
    model.load_state_dict(state_dict)

    test_dataset = NST_Dataset(content, style)
    test_loader = DataLoader(test_dataset, batch_size = 1, shuffle = False, num_workers=0, pin_memory=True)

    content, style = next(iter(test_loader))
    alpha = percent/100
    model.eval()
    with torch.no_grad():
      output = model.sample(content, style, alpha)
    bio = tensortophoto(output)

    return bio
