from bigmodule import I

import structlog

# metadata
author = "BigQuant"
category = r"深度学习\Keras\Reshaping"
friendly_name = "ZeroPadding1D - Keras"
doc_url = "https://keras.io/api/layers/reshaping_layers/zero_padding1d/"
cacheable = False

logger = structlog.get_logger()

def _none(x):
    if x == "None":
        return None
    return x

def run(
    padding: I.str("Padding，可以是1个整数或者由2个整数组成的元组") = "1",  # type: ignore
    debug: I.bool("输出调试信息，输出更多日志用于debug") = False,  # type: ignore
    input_layer: I.port("输入", optional=True) = None,  # type: ignore
) -> [I.port("layer", "data")]:  # type: ignore
    """ZeroPadding1D 是 Keras 中用于对1D输入（例如时间序列）进行零填充的类。"""
    
    import keras

    init_params = dict(
        padding=eval(padding),
    )
    call_params = dict()
    if input_layer is not None:
        call_params["inputs"] = input_layer
    if user_func is not None:
        user_func = user_func()
        if user_func:
            init_params.update(user_func.get("init", {}))
            call_params.update(user_func.get("call", {}))

    if debug:
        logger.info(f"{init_params=}, {call_params=}")

    layer = keras.layers.ZeroPadding1D(**init_params)(**call_params)

    return I.Outputs(data=layer)

def post_run(outputs):
    """后置运行函数"""
    return outputs