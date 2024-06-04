from bigmodule import I

import structlog

# metadata
author = "BigQuant"
category = r"深度学习\Keras\Reshaping"
friendly_name = "Permute - Keras"
doc_url = "https://keras.io/api/layers/reshaping_layers/permute/"
cacheable = False

logger = structlog.get_logger()

USER_PARAMS_DESC = """自定义参数"""
USER_PARAMS = """def bigquant_run():
    # https://keras.io/api/layers/reshaping_layers/permute/

    # import keras

    init_params = dict(
        # dims,
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
    dims: I.str("维度置换模式，不包括批量维度。索引从1开始，例如 (2, 1) 将置换输入的第一和第二维度。"),  # type: ignore
    user_params: I.code(USER_PARAMS_DESC, I.code_python, USER_PARAMS, specific_type_name="函数", auto_complete_type="python") = None,  # type: ignore
    debug: I.bool("输出调试信息，输出更多日志用于debug") = False,  # type: ignore
    input_layer: I.port("输入", optional=True) = None,  # type: ignore
) -> [I.port("layer", "data")]:  # type: ignore
    """Permute 是 Keras 中用于根据给定模式置换输入维度的类。"""

    import keras

    # 将 dims 参数从字符串转换为元组
    dims_tuple = tuple(map(int, dims.strip("()").split(",")))

    init_params = dict(
        dims=dims_tuple,
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

    layer = keras.layers.Permute(**init_params)(**call_params)

    return I.Outputs(data=layer)


def post_run(outputs):
    """后置运行函数"""
    return outputs