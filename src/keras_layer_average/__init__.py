from bigmodule import I

import structlog

# metadata
author = "BigQuant"
category = r"深度学习\Keras\Merging"
friendly_name = "Average - Keras"
doc_url = "https://keras.io/api/layers/merging_layers/average/"
cacheable = False

logger = structlog.get_logger()

USER_PARAMS_DESC = """自定义参数"""
USER_PARAMS = """def bigquant_run():
    # https://keras.io/api/layers/merging_layers/average/

    # import keras

    init_params = dict(
        # **kwargs
    )
    call_params = dict(
        # inputs,
        # **kwargs
    )

    return {"init": init_params, "call": call_params}
"""


def run(
    user_params: I.code(USER_PARAMS_DESC, I.code_python, USER_PARAMS, specific_type_name="函数", auto_complete_type="python") = None,  # type: ignore
    debug: I.bool("输出调试信息，输出更多日志用于debug") = False,  # type: ignore
    input_layers_1: I.port("输入1", optional=False) = None,  # type: ignore
    input_layers_2: I.port("输入2", optional=False) = None,  # type: ignore
    input_layers_3: I.port("输入3", optional=True) = None,  # type: ignore
) -> [I.port("layer", "data")]:  # type: ignore
    """Average 层是 Keras 中用于对一组输入张量进行元素级平均的类。"""

    import keras

    init_params = dict()
    call_params = dict()
    input_list = []
    if input_layers_1 is not None:
        input_list.append(input_layers_1)
    if input_layers_2 is not None:
        input_list.append(input_layers_2)
    if input_layers_3 is not None:
        input_list.append(input_layers_3)

    call_params["inputs"] = input_list

    if user_params is not None:
        user_params = user_params()
        if user_params:
            init_params.update(user_params.get("init", {}))
            call_params.update(user_params.get("call", {}))

    if debug:
        logger.info(f"{init_params=}, {call_params=}")

    layer = keras.layers.Average(**init_params)(**call_params)

    return I.Outputs(data=layer)


def post_run(outputs):
    """后置运行函数"""
    return outputs