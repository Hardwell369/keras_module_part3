from bigmodule import I

import structlog

# metadata
author = "BigQuant"
category = r"深度学习\Keras\Merging"
friendly_name = "Subtract - Keras"
doc_url = "https://keras.io/api/layers/merging_layers/subtract/"
cacheable = False

logger = structlog.get_logger()

USER_PARAMS_DESC = """自定义参数"""
USER_PARAMS = """def bigquant_run():
    # https://keras.io/api/layers/merging_layers/subtract/

    # import keras

    init_params = dict(
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
    user_params: I.code(USER_PARAMS_DESC, I.code_python, USER_PARAMS, specific_type_name="函数", auto_complete_type="python") = None,  # type: ignore
    debug: I.bool("输出调试信息，输出更多日志用于debug") = False,  # type: ignore
    input_layer_1: I.port("输入1", optional=False) = None,  # type: ignore
    input_layer_2: I.port("输入2", optional=False) = None,  # type: ignore
) -> [I.port("layer", "data")]:  # type: ignore
    """Subtract 是 Keras 中用于执行元素级减法的层。它接受两个形状相同的张量列表作为输入，并返回一个形状相同的单一张量(inputs[0] - inputs[1])。"""

    import keras

    init_params = dict()
    call_params = dict()
    if input_layer_1 is not None and input_layer_2 is not None:
        call_params["inputs"] = [input_layer_1, input_layer_2]
    else :
        raise ValueError("input_layer_1 and input_layer_2 should not be None")

    if user_params is not None:
        user_params = user_params()
        if user_params:
            init_params.update(user_params.get("init", {}))
            call_params.update(user_params.get("call", {}))

    if debug:
        logger.info(f"{init_params=}, {call_params=}")

    layer = keras.layers.Subtract(**init_params)(**call_params)

    return I.Outputs(data=layer)


def post_run(outputs):
    """后置运行函数"""
    return outputs