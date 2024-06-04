from bigmodule import I

import structlog

# metadata
author = "BigQuant"
category = r"深度学习\Keras\Merging"
friendly_name = "Dot - Keras"
doc_url = "https://keras.io/api/layers/merging_layers/dot/"
cacheable = False

logger = structlog.get_logger()

USER_PARAMS_DESC = """自定义参数"""
USER_PARAMS = """def bigquant_run():
    # https://keras.io/api/layers/merging_layers/dot/
    
    # import keras

    init_params = dict(
        # axes,
        # normalize=False,
        # **kwargs
    )
    call_params = dict(
        # inputs,
    )

    return {"init": init_params, "call": call_params}
"""


def _none(x):
    if x == "None":
        return None
    return x


def run(
    axes: I.str("轴，沿哪个轴进行点积，可以是整数或整数元组") = "1",  # type: ignore
    normalize: I.bool("L2正则化，是否在进行点积前对样本进行L2正则化") = False,  # type: ignore
    user_params: I.code(USER_PARAMS_DESC, I.code_python, USER_PARAMS, specific_type_name="函数", auto_complete_type="python") = None,  # type: ignore
    debug: I.bool("输出调试信息，输出更多日志用于debug") = False,  # type: ignore
    input_layer_1: I.port("第一个输入", optional=False) = None,  # type: ignore
    input_layer_2: I.port("第二个输入", optional=False) = None,  # type: ignore
) -> [I.port("layer", "data")]:  # type: ignore
    """Dot 是 Keras 中用于计算两个张量点积的层。"""

    import keras

    init_params = dict(
        axes=eval(axes),
        normalize=normalize,
    )
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

    layer = keras.layers.Dot(**init_params)(**call_params)

    return I.Outputs(data=layer)


def post_run(outputs):
    """后置运行函数"""
    return outputs