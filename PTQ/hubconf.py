from models.resnet import resnet18 as _resnet18
from models.mobilenetv2 import mobilenetv2 as _mobilenetv2
from torch.hub import load_state_dict_from_url

dependencies = ['torch']

def resnet18(pretrained=False, **kwargs):
    # Call the model, load pretrained weights
    model = _resnet18(**kwargs)
    if pretrained:
        load_url = 'https://github.com/yhhhli/BRECQ/releases/download/v1.0/resnet18_imagenet.pth.tar'
        checkpoint = load_state_dict_from_url(url=load_url, map_location='cpu', progress=True)
        model.load_state_dict(checkpoint)
    return model

def mobilenetv2(pretrained=False, **kwargs):
    # Call the model, load pretrained weights
    model = _mobilenetv2(**kwargs)
    if pretrained:
        load_url = 'https://github.com/yhhhli/BRECQ/releases/download/v1.0/mobilenetv2.pth.tar'
        checkpoint = load_state_dict_from_url(url=load_url, map_location='cpu', progress=True)
        model.load_state_dict(checkpoint['model'])
    return model


