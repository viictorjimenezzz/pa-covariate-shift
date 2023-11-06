import os.path

import torch
from robustbench.utils import download_gdrive
from .dag_module import DAGModule
from .net import ConvMedBig
from .bpda import BPDAWrapper
from .jpeg_compression import JpegCompression, Identity
from .reverse_sigmoid import ReverseSigmoid

MODEL_ID = "1ZwcLum_-iesa6kWYJ--9HDH9m8xn5mDK"


def load_undefended_model():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    identity_layer = Identity()
    rs_defense = ReverseSigmoid()
    network1 = ConvMedBig(device=device, dataset='cifar10', width1=4, width2=4, width3=4, linear_size=200,
                          input_channel=3, with_normalization=True)
    model_path = os.path.join(os.path.dirname(__file__), 'model.pth')
    if not os.path.exists(model_path):
        download_gdrive(MODEL_ID, model_path)
    state_dict = torch.load(model_path, map_location=device)
    classifier = DAGModule([identity_layer, network1, rs_defense], device=device)
    classifier.load_state_dict(state_dict)
    classifier.eval()
    return classifier


def load_model():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    jpeg_defense = JpegCompression((0, 1), 80)
    rs_defense = ReverseSigmoid()
    network1 = ConvMedBig(device=device, dataset='cifar10', width1=4, width2=4, width3=4, linear_size=200,
                          input_channel=3, with_normalization=True)
    model_path = os.path.join(os.path.dirname(__file__), 'model.pth')
    if not os.path.exists(model_path):
        download_gdrive(MODEL_ID, model_path)
    state_dict = torch.load(model_path, map_location=device)
    classifier = DAGModule([jpeg_defense, network1, rs_defense], device=device)
    classifier.load_state_dict(state_dict)
    classifier.eval()
    return classifier


def load_bpda_model(jpeg_quality):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    jpeg_layer = JpegCompression((0, 1), quality=jpeg_quality)
    bpda_layer = BPDAWrapper(jpeg_layer)
    rs_defense = ReverseSigmoid()
    network1 = ConvMedBig(device=device, dataset='cifar10', width1=4, width2=4, width3=4, linear_size=200,
                          input_channel=3, with_normalization=True)
    model_path = os.path.join(os.path.dirname(__file__), 'model.pth')
    if not os.path.exists(model_path):
        download_gdrive(MODEL_ID, model_path)
    state_dict = torch.load(model_path, map_location=device)
    classifier = DAGModule([bpda_layer, network1, rs_defense], device=device)
    classifier.load_state_dict(state_dict)
    classifier.eval()
    return classifier


def load_bpda_custom_model(model, jpeg_quality=80):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    jpeg_layer = JpegCompression((0, 1), jpeg_quality)
    bpda_layer = BPDAWrapper(jpeg_layer)
    rs_defense = ReverseSigmoid()

    classifier = DAGModule([bpda_layer, model, rs_defense], device=device)
    classifier.eval()
    return classifier
