def make_model(conf, *args, **kwargs):
    model_type = conf.get_string("type", "sngp")  # single
    if model_type == "pixelnerf":
        from .models import PixelNeRFNet

        net = PixelNeRFNet(conf, *args, **kwargs)
    elif model_type == "sngp":
        from .models_sngp import SphereNGPRadianceField

        net = SphereNGPRadianceField()
    elif model_type == "ngp":
        from .models_ngp import NGPRadianceField

        net = NGPRadianceField()
    else:
        raise NotImplementedError("Unsupported model type", model_type)
    return net
