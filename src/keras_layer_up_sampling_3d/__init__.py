from bigmodule import I

import structlog

# metadata
author = "BigQuant"
category = r"深度学习\Keras\Reshaping"
friendly_name = "UpSampling3D - Keras"
doc_url = "https://keras.io/api/layers/reshaping_layers/up_sampling3d/"
cacheable = False

logger = structlog.get_logger()

DATA_FORMATS = ["channels_last", "channels_first"]

USER_PARAMS_DESC = """自定义参数"""
USER_PARAMS = """def bigquant_run():
    # https://keras.io/api/layers/reshaping_layers/up_sampling3d/

    # import keras

    init_params = dict(
        # size=(2, 2, 2),
        # data_format=None,
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
    size: I.str("上采样因子，为一个整数或3个整数的元组") = "(2, 2, 2)",  # type: ignore
    data_format: I.choice("数据格式，用于指定输入数据的维度顺序", DATA_FORMATS) = "channels_last",  # type: ignore
    user_params: I.code(USER_PARAMS_DESC, I.code_python, USER_PARAMS, specific_type_name="函数", auto_complete_type="python") = None,  # type: ignore
    debug: I.bool("输出调试信息，输出更多日志用于debug") = False,  # type: ignore
    input_layer: I.port("输入", optional=True) = None,  # type: ignore
) -> [I.port("layer", "data")]:  # type: ignore
    """UpSampling3D 是 Keras 中用于对 3D 输入进行上采样的层。"""
    
    import keras

    init_params = dict(
        size=eval(size),
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

    layer = keras.layers.UpSampling3D(**init_params)(**call_params)

    return I.Outputs(data=layer)


def post_run(outputs):
    """后置运行函数"""
    return outputs