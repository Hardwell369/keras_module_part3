from bigmodule import I

import structlog

# metadata
author = "BigQuant"
category = r"深度学习\Keras\Activation"
friendly_name = "Softmax - Keras"
doc_url = "https://keras.io/api/layers/activation_layers/softmax/"
cacheable = False

logger = structlog.get_logger()

USER_PARAMS_DESC = """自定义参数"""
USER_PARAMS = """def bigquant_run():
    # https://keras.io/api/layers/activation_layers/softmax/

    # import keras

    init_params = dict(
        # axis=-1,
        # **kwargs
    )
    call_params = dict(
        # inputs,
        # mask=None,
    )

    return {"init": init_params, "call": call_params}
"""


def run(
    axis: I.str("沿着哪个或哪些轴应用softmax归一化。") = "-1",  # type: ignore
    user_params: I.code(USER_PARAMS_DESC, I.code_python, USER_PARAMS, specific_type_name="函数", auto_complete_type="python") = None,  # type: ignore
    debug: I.bool("输出调试信息，输出更多日志用于debug") = False,  # type: ignore
    input_layer: I.port("输入", optional=True) = None,  # type: ignore
) -> [I.port("layer", "data")]:  # type: ignore
    """Softmax Keras 中用于创建Softmax激活层的类。Softmax激活函数常用于分类任务中，将输入向量转换为概率分布。"""

    import keras

    init_params = dict(
        axis=eval(axis),
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

    layer = keras.layers.Softmax(**init_params)(**call_params)

    return I.Outputs(data=layer)


def post_run(outputs):
    """后置运行函数"""
    return outputs