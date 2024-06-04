from bigmodule import I

import structlog

# metadata
author = "BigQuant"
category = r"深度学习\Keras\Reshaping"
friendly_name = "UpSampling2D - Keras"
doc_url = "https://keras.io/api/layers/reshaping_layers/up_sampling2d/"
cacheable = False

logger = structlog.get_logger()

INTERPOLATIONS = ["nearest", "bilinear", "bicubic", "lanczos3", "lanczos5"]
DATA_FORMATS = ["channels_last", "channels_first", "None"]

USER_PARAMS_DESC = """自定义函数，自定义设置层构建参数和调用参数"""
USER_PARAMS = """def bigquant_run():
    # https://keras.io/api/layers/reshaping_layers/up_sampling2d/

    # import keras

    init_params = dict(
        # size=(2, 2),
        # data_format=None,
        # interpolation="nearest",
        # **kwargs
    )
    call_params = dict(
        # inputs,
        # training=True,
        # mask=None,
    )

    return {"init": init_params, "call": call_params}
"""


def _none(value):
    return None if value == "None" else value


def run(
    size: I.str("上采样因子，可以是一个整数，或者是一个包含两个整数的元组。") = "(2, 2)",  # type: ignore
    data_format: I.choice("数据格式，'channels_last' 或 'channels_first'", DATA_FORMATS) = "channels_last",  # type: ignore
    interpolation: I.choice("插值方法，'nearest'、'bilinear'、'bicubic'、'lanczos3' 或 'lanczos5'", INTERPOLATIONS) = "nearest",  # type: ignore
    user_params: I.code(USER_PARAMS_DESC, I.code_python, USER_PARAMS, specific_type_name="函数", auto_complete_type="python") = None,  # type: ignore
    debug: I.bool("输出调试信息，输出更多日志用于debug") = False,  # type: ignore
    input_layer: I.port("输入", optional=True) = None,  # type: ignore
) -> [I.port("layer", "data")]:  # type: ignore
    """UpSampling2D 是 Keras 中用于对 2D 输入进行上采样的层。上采样层通过插值方法来调整输入数据的大小。"""

    import keras

    init_params = dict(
        size=eval(size),
        data_format=data_format,
        interpolation=_none(interpolation),
    )
    call_params = dict()
    if input_layer is not None:
        call_params["inputs"] = input_layer
    if user_params is not None:
        user_params = user_params()
        if user_params:
            init_params.update(user_params.get("init", {}))
            call_params.update(user_params.get("call", {}))

    if debug:
        logger.info(f"{init_params=}, {call_params=}")

    layer = keras.layers.UpSampling2D(**init_params)(**call_params)

    return I.Outputs(data=layer)

def post_run(outputs):
    """后置运行函数"""
    return outputs