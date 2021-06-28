from PIL import Image
from torchvision import transforms
from torch import Tensor


def image_loader(image_path: str, img_size: tuple = (128, 128)) -> Tensor:
    """
    Loading and preprocessing an image
    :param image_path: path to image
    :param img_size: image size
    :return: image format torch.Tensor
    """
    loader: transforms.Compose = transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor()])
    image: Image = Image.open(image_path)
    img_tensor: Tensor = loader(image).unsqueeze(0)

    return img_tensor


def save_img(img_tensor: Tensor, path: str) -> None:
    """
    Image save function
    :param img_tensor: image format torch.Tensor
    :param path: path to image
    :return:
    """
    tensor_to_pil: transforms.ToPILImage = transforms.ToPILImage()
    img_tensor: Tensor = img_tensor.squeeze(0)
    img: Image = tensor_to_pil(img_tensor)
    img.save(path)
