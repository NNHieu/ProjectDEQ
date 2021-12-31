from .feed_forward_net import ff_net
from .fixed_point_net import fp_net
from .recurrent_dilated_net import recur_dilated_net
from .recurrent_injected_net import recur_injected_net
from .recurrent_net import recur_net


def get_model(func_model, arch: dict):
    """Function to load the model object
    input:
        model:      str, Name of the model
        width:      int, Width of network
        depth:      int, Depth of network
    return:
        net:        Pytorch Network Object
    """
    net = eval(func_model.lower())(depth=arch["depth"], width=arch["width"], cfg=arch)
    return net
