from bigmodule import I

import structlog

# metadata
author = "BigQuant"
category = r"深度学习\Keras\Activation"
friendly_name = "LeakyReLU - Keras"
doc_url = "https://keras.io/api/layers/activation_layers/leaky_relu/"
cacheable = False

logger = structlog.get_logger()

USER_PARAMS_DESC = """自定义参数"""
USER_PARAMS = """def bigquant_run():
    # https://keras.io/api/layers/activation_layers/leaky_relu/

    # import keras

    init_params = dict(
        # negative_slope=0.3,
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
    negative_slope: I.float("负斜率系数，当输入小于0时，输出值为负斜率乘以输入值。", min=0.0) = 0.3,  # type: ignore
    user_params: I.code(USER_PARAMS_DESC, I.code_python, USER_PARAMS, specific_type_name="函数", auto_complete_type="python") = None,  # type: ignore
    debug: I.bool("输出调试信息，输出更多日志用于debug") = False,  # type: ignore
    input_layer: I.port("输入", optional=True) = None,  # type: ignore
) -> [I.port("layer", "data")]:  # type: ignore
    """LeakyReLU 是 Keras 中用于创建带有负斜率的修正线性单元激活层的类。"""

    import keras

    init_params = dict(
        negative_slope=negative_slope,
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

    layer = keras.layers.LeakyReLU(**init_params)(**call_params)

    return I.Outputs(data=layer)


def post_run(outputs):
    """后置运行函数"""
    return outputs