from bigmodule import I

import structlog

# metadata
author = "BigQuant"
category = r"深度学习\Keras\Activation"
friendly_name = "ReLU - Keras"
doc_url = "https://keras.io/api/layers/activation_layers/relu/"
cacheable = False

logger = structlog.get_logger()

USER_PARAMS_DESC = """自定义参数"""
USER_PARAMS = """def bigquant_run():
    # https://keras.io/api/layers/activation_layers/relu/

    # import keras

    init_params = dict(
        # max_value=None,
        # negative_slope=0.0,
        # threshold=0.0,
        # **kwargs
    )
    call_params = dict(
        # inputs
    )

    return {"init": init_params, "call": call_params}
"""

def run(
    max_value: I.float("最大激活值，None 表示无限制。", min=0.0) = None,  # type: ignore
    negative_slope: I.float("负斜率系数。", min=0.0) = 0.0,  # type: ignore
    threshold: I.float("阈值，用于阈值激活。", min=0.0) = 0.0,  # type: ignore
    user_params: I.code(USER_PARAMS_DESC, I.code_python, USER_PARAMS, specific_type_name="函数", auto_complete_type="python") = None,  # type: ignore
    debug: I.bool("输出调试信息，输出更多日志用于debug") = False,  # type: ignore
    input_layer: I.port("输入", optional=True) = None,  # type: ignore
) -> [I.port("layer", "data")]:  # type: ignore
    """ReLU 是 Keras 中用于创建修正线性单元激活函数层的类。"""

    import keras

    init_params = dict(
        max_value=max_value,
        negative_slope=negative_slope,
        threshold=threshold,
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

    layer = keras.layers.ReLU(**init_params)(**call_params)

    return I.Outputs(data=layer)


def post_run(outputs):
    """后置运行函数"""
    return outputs