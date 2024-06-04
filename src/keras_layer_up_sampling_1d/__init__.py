from bigmodule import I
import structlog

# metadata
author = "BigQuant"
category = r"深度学习\Keras\Reshaping"
friendly_name = "UpSampling1D - Keras"
doc_url = "https://keras.io/api/layers/reshaping_layers/up_sampling1d/"
cacheable = False

logger = structlog.get_logger()

USER_PARAMS_DESC = """自定义参数"""
USER_PARAMS = """def bigquant_run(
    # https://keras.io/api/layers/reshaping_layers/up_sampling1d/

    # import keras

    init_params = dict(
        # size,
        # **kwargs
    )
    call_params = dict(
        # inputs,
    )

    return {"init": init_params, "call": call_params}
"""

def run(
    size: I.int("上采样因子", min=1) = 2,  # type: ignore
    user_params: I.code(USER_PARAMS_DESC, I.code_python, USER_PARAMS, specific_type_name="函数", auto_complete_type="python") = None,  # type: ignore
    debug: I.bool("输出调试信息，输出更多日志用于debug") = False,  # type: ignore
    input_layer: I.port("输入", optional=True) = None,  # type: ignore
) -> [I.port("layer", "data")]:  # type: ignore
    """UpSampling1D 是 Keras 中用于对 1D 输入进行上采样的层。它会沿时间轴重复每个时间步 size 次。"""

    import keras

    init_params = dict(
        size=size,
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

    layer = keras.layers.UpSampling1D(**init_params)(**call_params)

    return I.Outputs(data=layer)


def post_run(outputs):
    """后置运行函数"""
    return outputs