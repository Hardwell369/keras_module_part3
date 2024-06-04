from bigmodule import I

import structlog

# metadata
author = "BigQuant"
category = r"深度学习\Keras\Reshaping"
friendly_name = "Cropping3D - Keras"
doc_url = "https://keras.io/api/layers/reshaping_layers/cropping3d/"
cacheable = False

logger = structlog.get_logger()

DATA_FORMATS = ["channels_last", "channels_first"]
USER_PARAMS_DESC = """自定义参数"""
USER_PARAMS = """def bigquant_run():
    # https://keras.io/api/layers/reshaping_layers/cropping3d/

    # import keras

    init_params = dict(
        # cropping=((1, 1), (1, 1), (1, 1)),
        # data_format=None,
        # **kwargs
    )
    call_params = dict(
        # inputs,
        # training=True,
        # mask=None,
    )

    return {"init": init_params, "call": call_params}
"""


def run(
    cropping: I.str("裁剪参数，Int或3个Int的tuple或3个包含2个Int的tuple。") = "((1, 1), (1, 1), (1, 1))",  # type: ignore
    data_format: I.choice("数据格式，channels_last 或 channels_first", DATA_FORMATS) = "channels_last",  # type: ignore
    user_params: I.code(USER_PARAMS_DESC, I.code_python, USER_PARAMS, specific_type_name="函数", auto_complete_type="python") = None,  # type: ignore
    debug: I.bool("输出调试信息，输出更多日志用于debug") = False,  # type: ignore
    input_layer: I.port("输入", optional=True) = None,  # type: ignore
) -> [I.port("layer", "data")]:  # type: ignore
    """Cropping3D 是 Keras 中用于3D数据（例如空间或时空数据）的裁剪层。"""

    import keras

    init_params = dict(
        cropping=eval(cropping),
        data_format=data_format,
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

    layer = keras.layers.Cropping3D(**init_params)(**call_params)

    return I.Outputs(data=layer)


def post_run(outputs):
    """后置运行函数"""
    return outputs