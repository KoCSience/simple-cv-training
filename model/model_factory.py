import os

from simple_cv_core.registry import get_model, register_model

from model import (
    ClassificationBaseModel,
    ModelConfig,
    ResNet50,
    ResNet18,
    ABNResNet50,
    X3DM,
    ViTb,
    ZeroOutputModel,
)


register_model("resnet18")(ResNet18)
register_model("resnet50")(ResNet50)
register_model("abn_r50")(ABNResNet50)
register_model("vit_b")(ViTb)
register_model("x3d")(X3DM)
register_model("zero_output_dummy")(ZeroOutputModel)


def set_torch_home(
    model_info: ModelConfig
) -> None:
    """Specity the directory where a pre-trained model is stored.
    Otherwise, by default, models are stored in users home dir `~/.torch`
    """
    os.environ['TORCH_HOME'] = model_info.torch_home


def configure_model(
        model_info: ModelConfig
) -> ClassificationBaseModel:
    """model factory

    model_info:
        model_info (ModelInfo): information for model

    Raises:
        ValueError: invalide model name given by command line

    Returns:
        ClassificationBaseModel: model
    """

    if model_info.use_pretrained:
        set_torch_home(model_info)

    model_builder = get_model(model_info.model_name)
    return model_builder(model_info)  # type: ignore[operator, no-any-return]
