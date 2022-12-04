from typing import Any, Dict

import torch
from torch import nn
import torchvision
import operator as op 
from torchvision import models 


class VisualBackbone(nn.Module):
    r"""
    Base class for all visual backbones. All child classes can simply inherit
    from :class:`~torch.nn.Module`, however this is kept here for uniform
    type annotations.
    """

    def __init__(self, visual_feature_size: int):
        super().__init__()
        self.visual_feature_size = visual_feature_size


class TorchvisionVisualBackbone(VisualBackbone):
    r"""
    A visual backbone from `Torchvision model zoo
    <https://pytorch.org/docs/stable/torchvision/models.html>`_. Any model can
    be specified using corresponding method name from the model zoo.

    Args:
        name: Name of the model from Torchvision model zoo.
        visual_feature_size: Size of the channel dimension of output visual
            features from forward pass.
        pretrained: Whether to load ImageNet pretrained weights from Torchvision.
        frozen: Whether to keep all weights frozen during training.
    """

    def __init__(
        self,
        name: str = "resnet50",
        visual_feature_size: int = 2048,
        pretrained: bool = False,
        frozen: bool = False,
    ):
        super().__init__(visual_feature_size)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        model_name = "resnet152.th"
        real_name, _ = model_name.split('.')
        endpoint = op.attrgetter(real_name)(models) # gọi models.resnet152()
        if endpoint is not None:
            features_extractor = endpoint(pretrained=True, progress=True)
            features_extractor = nn.Sequential(*list(features_extractor.children())[:-2])
            for prm in features_extractor.parameters():
                prm.requires_grad = False
        
        self.cnn = features_extractor
        self.cnn.eval()

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        r"""
        Compute visual features for a batch of input images.

        Args:
            image: Batch of input images. A tensor of shape ``(batch_size, 3,
                height, width)``.

        Returns:
            A tensor of shape ``(batch_size, channels, height, width)``, for
            example it will be ``(batch_size, 2048, 7, 7)`` for ResNet-50.
            
            vit (batch_size,49,512)
        """
        
        # with torch.no_grad():
        #     print(image.shape)
        #     features = self.cnn(image[None,...].to(self.device))
            # embedding = torch.flatten(features, start_dim=1).T.cpu().numpy()  # 49, 2048
        return image

    def detectron2_backbone_state_dict(self) -> Dict[str, Any]:
        r"""
        Return state dict of visual backbone which can be loaded with
        `Detectron2 <https://github.com/facebookresearch/detectron2>`_.
        This is useful for downstream tasks based on Detectron2 (such as
        object detection and instance segmentation). This method renames
        certain parameters from Torchvision-style to Detectron2-style.

        Returns:
            A dict with three keys: ``{"model", "author", "matching_heuristics"}``.
            These are necessary keys for loading this state dict properly with
            Detectron2.
        """
        # Detectron2 backbones have slightly different module names, this mapping
        # lists substrings of module names required to be renamed for loading a
        # torchvision model into Detectron2.
        DETECTRON2_RENAME_MAPPING: Dict[str, str] = {
            "layer1": "res2",
            "layer2": "res3",
            "layer3": "res4",
            "layer4": "res5",
            "bn1": "conv1.norm",
            "bn2": "conv2.norm",
            "bn3": "conv3.norm",
            "downsample.0": "shortcut",
            "downsample.1": "shortcut.norm",
        }
        # Populate this dict by renaming module names.
        d2_backbone_dict: Dict[str, torch.Tensor] = {}

        for name, param in self.cnn.state_dict().items():
            for old, new in DETECTRON2_RENAME_MAPPING.items():
                name = name.replace(old, new)

            # First conv and bn module parameters are prefixed with "stem.".
            if not name.startswith("res"):
                name = f"stem.{name}"

            d2_backbone_dict[name] = param

        return {
            "model": d2_backbone_dict,
            "__author__": "Karan Desai",
            "matching_heuristics": True,
        }
