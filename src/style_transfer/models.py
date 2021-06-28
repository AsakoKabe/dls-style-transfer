from typing import Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor
from torch.nn import Sequential

from .utils import image_loader


class ContentLoss(nn.Module):
    """
    Content loss layer
    """

    def __init__(self, target: Tensor):
        super(ContentLoss, self).__init__()
        # we 'detach' the target content from the tree used
        # to dynamically compute the gradient: this is a stated value,
        # not a variable. Otherwise the forward method of the criterion
        # will throw an error.
        self.__target: Tensor = target.detach()  # это константа. Убираем ее из дерева вычеслений
        self.__loss: Tensor = F.mse_loss(self.__target, self.__target)  # to initialize with something

    def forward(self, features: Tensor) -> Tensor:
        self.__loss: Tensor = F.mse_loss(features, self.__target)

        return features

    def get_loss(self) -> Tensor:
        return self.__loss


class StyleLoss(nn.Module):
    """
    Style Loss layer
    """

    def __init__(self, target_feature: Tensor):
        super(StyleLoss, self).__init__()
        self.__target: Tensor = self.gram_matrix(target_feature).detach()
        self.__loss: Tensor = F.mse_loss(self.__target, self.__target)  # to initialize with something

    @staticmethod
    def gram_matrix(features_image: Tensor) -> Tensor:
        """
        Computes gramm matrix
        :param features_image: features image
        :return: gramm matrix
        """
        batch_size, h, w, f_map_num = features_image.size()  # batch size(=1)
        # b=number of feature maps
        # (h,w)=dimensions of a feature map (N=h*w)
        features: Tensor = features_image.view(batch_size * h, w * f_map_num)  # resise F_XL into \hat F_XL
        G: Tensor = torch.mm(features, features.t())  # compute the gram product
        # we 'normalize' the values of the gram matrix
        # by dividing by the number of element in each feature maps.
        return G.div(batch_size * h * w * f_map_num)

    def forward(self, features: Tensor) -> Tensor:
        g: Tensor = self.gram_matrix(features)
        self.__loss: Tensor = F.mse_loss(g, self.__target)

        return features

    def get_loss(self):
        return self.__loss


class Normalization(nn.Module):
    """
    Normalizing images for VGG19
    """

    def __init__(self):
        super(Normalization, self).__init__()
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        self.__mean: Tensor = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
        self.__std: Tensor = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)

    def forward(self, img: Tensor) -> Tensor:
        # normalize img
        return (img - self.__mean) / self.__std


class VGG(nn.Module):
    """
    VGG19 which contains only first 5 convolutions
    """

    def __init__(self):
        super(VGG, self).__init__()
        self.__features: nn.Sequential = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
        )

    def get_features(self) -> nn.Sequential:
        return self.__features

    def forward(self, x) -> Tensor:
        x = self.__features(x)

        return x


class StyleTransfer:
    """
    Style transfer image with VGG19
    """

    def __init__(self):
        self.__content_img = None
        # We store the path to the style image so that when the output image is resized, the style image will be resized
        self.__img_size = (128, 128)
        self.__style_img_path = 'images/picasso.jpg'
        self.__style_img = image_loader(self.__style_img_path, self.__img_size)
        self.__input_img = None
        self.__content_layers = ['conv_4']
        self.__style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
        self.__cnn = VGG().get_features()
        self.__cnn.load_state_dict(torch.load('style_transfer/weights/VGG19.pth'))
        # on/off
        self.__mode = True

    def set_content_img(self, content_image_path: str) -> None:
        self.__content_img: Tensor = image_loader(content_image_path, self.__img_size)
        self.__input_img: Tensor = self.__content_img.clone()

    def set_style_img(self, style_img_path: str) -> None:
        self.__style_img: Tensor = image_loader(style_img_path, self.__img_size)

    def set_style_img_path(self, style_img_path: str) -> None:
        self.__style_img_path: str = style_img_path

    def get_style_img_path(self) -> str:
        return self.__style_img_path

    def get_style_img(self) -> Tensor:
        return self.__style_img

    def set_img_size(self, new_img_size: Tuple[int]) -> None:
        self.__img_size = new_img_size

    def change_mode(self) -> None:
        self.__mode = False if self.__mode else True

    def get_mode(self) -> bool:
        return self.__mode

    def get_style_model_and_losses(self) -> Tuple[Sequential, List[StyleLoss], List[ContentLoss]]:
        # just in order to have an iterable access to or list of content/syle
        # losses
        content_losses = []
        style_losses = []

        # assuming that cnn is a nn.Sequential, so we make a new nn.Sequential
        # to put in modules that are supposed to be activated sequentially
        model = nn.Sequential(Normalization())

        i = 0  # increment every time we see a conv
        for layer in self.__cnn.children():
            if isinstance(layer, nn.Conv2d):
                i += 1
                name = 'conv_{}'.format(i)
            elif isinstance(layer, nn.ReLU):
                name = 'relu_{}'.format(i)
                # The in-place version doesn't play very nicely with the ContentLoss
                # and StyleLoss we insert below. So we replace with out-of-place
                # ones here.
                # Переопределим relu уровень
                layer = nn.ReLU(inplace=False)
            elif isinstance(layer, nn.MaxPool2d):
                name = 'pool_{}'.format(i)
            elif isinstance(layer, nn.BatchNorm2d):
                name = 'bn_{}'.format(i)
            else:
                raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

            model.add_module(name, layer)

            if name in self.__content_layers:
                # add content loss:
                target = model(self.__content_img).detach()
                content_loss = ContentLoss(target)
                model.add_module("content_loss_{}".format(i), content_loss)
                content_losses.append(content_loss)

            if name in self.__style_layers:
                # add style loss:
                target_feature = model(self.__style_img).detach()
                style_loss = StyleLoss(target_feature)
                model.add_module("style_loss_{}".format(i), style_loss)
                style_losses.append(style_loss)

        # now we trim off the layers after the last content and style losses
        # выбрасываем все уровни после последенего style loss или content loss
        for i in range(len(model) - 1, -1, -1):
            if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
                break

        model = model[:(i + 1)]

        return model, style_losses, content_losses

    def fit(self, num_steps=500, style_weight=100000, content_weight=1) -> Tensor:
        """Run the style transfer."""
        model, style_losses, content_losses = self.get_style_model_and_losses()
        optimizer = optim.LBFGS([self.__input_img.requires_grad_()])

        run = [0]
        while run[0] <= num_steps:
            def closure():
                # correct the values
                # это для того, чтобы значения тензора картинки не выходили за пределы [0;1]
                self.__input_img.data.clamp_(0, 1)
                optimizer.zero_grad()
                model(self.__input_img)

                style_score = 0
                content_score = 0

                for sl in style_losses:
                    style_score += sl.get_loss()
                for cl in content_losses:
                    content_score += cl.get_loss()

                # взвешивание ощибки
                style_score *= style_weight
                content_score *= content_weight
                loss = style_score + content_score
                loss.backward()
                run[0] += 1
                # if run[0] % 50 == 0:
                #     print('Run {} : Style Loss = {:4f} Content Loss = {:4f}'.format(
                #         run[0], style_score.item(), content_score.item()))

                return style_score + content_score

            optimizer.step(closure)

        # a last correction...
        self.__input_img.data.clamp_(0, 1)

        return self.__input_img
