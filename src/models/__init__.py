from models.BranchedERFNet import BranchedERFNet
from models.BranchedBiSeNetV2_Custom import BranchedBiSeNetV2_Custom
from models.BranchedBiSeNetV2_Custom_WOD import BranchedBiSeNetV2_Custom_WOD
from models.BranchedBiSeNetV2_Custom_CA1 import BranchedBiSeNetV2_Custom_CA1


def get_model(name, model_opts):
    # same as original spatial embedding
    if name == "branched-erfnet":
        model = BranchedERFNet(**model_opts)
        return model

    # BiSeNetV2(with detail branch) encoder + feature fusion with BGA
    elif name == "branched-bisenetv2-custom":
        model = BranchedBiSeNetV2_Custom(**model_opts)
        return model

    # BiSeNetV2(with detail branch) encoder + feature fusion with cross-attention
    elif name == "branched-bisenetv2-custom_ca1":
        model = BranchedBiSeNetV2_Custom_CA1(**model_opts)
        return model

    # BiSeNetV2(wo detail branch) encoder
    elif name == "branched-bisenetv2-custom_wod":
        model = BranchedBiSeNetV2_Custom_WOD(**model_opts)
        return model

    else:
        raise RuntimeError("model \"{}\" not available".format(name))