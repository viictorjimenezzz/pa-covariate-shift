from typing import Optional

import torch
import torch.nn as nn
from copy import deepcopy

from pametric.lightning import SplitClassifier

class Wong2020_Split(SplitClassifier):
    def __init__(self, net: nn.Module):
        super().__init__(net, "foo")
        self.feature_extractor = nn.Sequential(
            net.conv1, net.layer1, net.layer2, net.layer3, net.layer4,
            nn.BatchNorm2d(512), nn.ReLU(), nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten()
        )
        self.classifier = net.linear

class Wang2023_Split(SplitClassifier):
    def __init__(self, net: nn.Module):
        super().__init__(net, "foo")
        self.feature_extractor = nn.Sequential(
            net.init_conv, net.layer, net.batchnorm, net.relu,
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        self.classifier = net.logits

class Engstrom2019_Split(SplitClassifier):
    def __init__(self, net: nn.Module):
        super().__init__(net, "foo")
        self.feature_extractor = nn.Sequential(
            *list(net.children())[:-1],
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        self.classifier = net.linear

class Addepalli2021_Split(SplitClassifier):
    def __init__(self, net: nn.Module):
        super().__init__(net, "foo")
        self.feature_extractor = nn.Sequential(
                *list(net.children())[:-1], 
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten()
        )
        self.classifier = net.linear

class BPDA_Split(SplitClassifier):
    def __init__(self, net: nn.Module):
        super().__init__(net, "foo") 
        split_layer_index = 9
        layers = list(net.eval_model.moduleList[1].blocks.layers)

        self.normalizer = net.eval_model.moduleList[1].normalizer
        self.features = nn.Sequential(*layers[:split_layer_index+1])
        self.feature_extractor = nn.Sequential(
            self.normalizer,
            self.features,
            # nn.AdaptiveAvgPool2d((1, 1)),
            # nn.Flatten()
        )

        self.reverse_sigmoid = net.eval_model.moduleList[2]
        self.classifier = nn.Sequential(
            *layers[split_layer_index+1:],
            self.reverse_sigmoid
        )
        

class MeasureOutputAdv:
    """
    Override the evaluation of the model to adjust it to the robustbench models.
    """

    def _get_model_to_eval(self, net: nn.Module, net_name: str) -> nn.Module:
        adv_net_name = self.net_name

        if adv_net_name == "BPDA":
            return BPDA_Split(net=deepcopy(net))
            
        elif adv_net_name == "Addepalli2021Towards_RN18":
            return Addepalli2021_Split(net=deepcopy(net))
        
        # There is no Linf attack for it
        # elif adv_net_name == "Peng2023Robust":
        #     import ipdb; ipdb.set_trace()

        elif adv_net_name == "Engstrom2019Robustness":
            return Engstrom2019_Split(net=deepcopy(net))

        elif adv_net_name == "Wang2023Better_WRN-28-10":
            return Wang2023_Split(net=deepcopy(net))

        elif adv_net_name == "Wong2020Fast":
            return Wong2020_Split(net=deepcopy(net))
        
        else: # Standard
            return SplitClassifier(net=deepcopy(net), net_name=net_name)
        

    
from src.callbacks.pa_distances import PAOutput_Callback

class PAOutput_Callback(MeasureOutputAdv, PAOutput_Callback):
    def __init__(self, *args, net_name: Optional[str] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.net_name = net_name

    
from pametric.lightning.callbacks import CentroidDistance_Callback, FrechetInceptionDistance_Callback, MMD_Callback

class CentroidDistance_Callback(MeasureOutputAdv, CentroidDistance_Callback):
    def __init__(self, *args, net_name: Optional[str] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.net_name = net_name

class FrechetInceptionDistance_Callback(MeasureOutputAdv, FrechetInceptionDistance_Callback):
    def __init__(self, *args, net_name: Optional[str] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.net_name = net_name
    
class MMD_Callback(MeasureOutputAdv, MMD_Callback):
    def __init__(self, *args, net_name: Optional[str] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.net_name = net_name