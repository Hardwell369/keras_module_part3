from bigmodule import I

import structlog

# metadata
author = "BigQuant"
category = r"深度学习\Keras\Merging"
friendly_name = "Concatenate - Keras"
doc_url = "https://keras.io/api/layers/merging_layers/concatenate/"
cacheable = False

logger = structlog.get_logger()

USER_PARAMS_DESC = """自定义参数"""
USER_PARAMS = """def bigquant_run():
    # https://keras.io/api/layers/merging_layers/concatenate/

    # import keras

    init_params = dict(
        # axis=-1,
        # **kwargs
    )
    call_params = dict(
        # inputs
    )

    return {"init": init_params, "call": call_params}
"""

def run(
    axis: I.int("沿着哪个轴进行拼接。") = -1,  # type: ignore
    user_params: I.code(USER_PARAMS_DESC, I.code_python, USER_PARAMS, specific_type_name="函数", auto_complete_type="python") = None,  # type: ignore
    debug: I.bool("输出调试信息，输出更多日志用于debug") = False,  # type: ignore
    input_layer_1: I.port("输入1", optional=False) = None,  # type: ignore
    input_layer_2: I.port("输入2", optional=False) = None,  # type: ignore
    input_layer_3: I.port("输入3", optional=True) = None,  # type: ignore
) -> [I.port("layer", "data")]:  # type: ignore
    """Concatenate 是 Keras 中用于拼接多个输入张量的层。"""

    import keras

    init_params = dict(
        axis=axis,
    )
    call_params = dict()
    input_list = []
    if input_layer_1 is not None:
        input_list.append(input_layer_1)
    if input_layer_2 is not None:
        input_list.append(input_layer_2)
    if input_layer_3 is not None:
        input_list.append(input_layer_3)

    call_params["inputs"] = input_list

    if user_params is not None:
        user_params = user_params()
        if user_params:
            init_params.update(user_params.get("init", {}))
            call_params.update(user_params.get("call", {}))

    if debug:
        logger.info(f"{init_params=}, {call_params=}")

    layer = keras.layers.Concatenate(**init_params)(**call_params)

    return I.Outputs(data=layer)

def post_run(outputs):
    """后置运行函数"""
    return outputs