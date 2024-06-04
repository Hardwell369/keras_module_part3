from bigmodule import I

import structlog

# metadata
author = "BigQuant"
category = r"深度学习\Keras\Reshaping"
friendly_name = "Flatten - Keras"
doc_url = "https://keras.io/api/layers/reshaping_layers/flatten/"
cacheable = False

logger = structlog.get_logger()

DATA_FORMATS = ["channels_last", "channels_first"]

USER_PARAMS_DESC = """自定义参数"""
USER_PARAMS = """def bigquant_run(
    # https://keras.io/api/layers/reshaping_layers/flatten/

    # import keras

    init_params = dict(
        # data_format="channels_last",
    )
    call_params = dict(
        # inputs,
    )

    return {"init": init_params, "call": call_params}
"""


def run(
    data_format: I.choice("输入数据的格式，data_format", values=DATA_FORMATS) = "channels_last",  # type: ignore
    input_layer: I.port("输入", optional=False) = None,  # type: ignore
    user_params: I.code(USER_PARAMS_DESC, I.code_python, USER_PARAMS, specific_type_name="函数", auto_complete_type="python") = None,  # type: ignore
    debug: I.bool("输出调试信息，输出更多日志用于debug") = False,  # type: ignore
) -> [I.port("layer", "data")]:  # type: ignore
    """Flatten 是 Keras 中用于将输入展平的层，不会影响批处理大小。"""

    import keras

    init_params = dict(
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

    layer = keras.layers.Flatten(**init_params)(**call_params)

    return I.Outputs(data=layer)


def post_run(outputs):
    """后置运行函数"""
    return outputs