from bigmodule import I
import structlog

# metadata
author = "BigQuant"
category = r"深度学习\Keras\Activation"
friendly_name = "PReLU - Keras"
doc_url = "https://keras.io/api/layers/activation_layers/prelu/"
cacheable = False

logger = structlog.get_logger()

INITIALIZERS = [
    "Constant",
    "GlorotNormal",
    "GlorotUniform",
    "HeNormal",
    "HeUniform",
    "Identity",
    "Initializer",
    "LecunNormal",
    "LecunUniform",
    "Ones",
    "OrthogonalInitializer",
    "RandomNormal",
    "RandomUniform",
    "TruncatedNormal",
    "VarianceScaling",
    "Zeros",
    "None",
]
REGULARIZERS = ["L1", "L2", "L1L2", "OrthogonalRegularizer", "Regularizer", "None"]
CONSTRAINTS = ["Constraint", "MaxNorm", "MinMaxNorm", "NonNeg", "UnitNorm", "None"]

USER_PARAMS_DESC = """自定义参数"""
USER_PARAMS = """def bigquant_run():
    # https://keras.io/api/layers/activation_layers/prelu/

    # import keras

    init_params = dict(
        # alpha_initializer="Zeros",
        # alpha_regularizer="None",
        # alpha_constraint="None",
        # shared_axes=None,
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
    alpha_initializer: I.choice("权重矩阵初始化方法，alpha_initializer", INITIALIZERS) = "Zeros",  # type: ignore
    alpha_regularizer: I.choice("权重矩阵正则化方法，alpha_regularizer", REGULARIZERS) = "None",  # type: ignore
    alpha_constraint: I.choice("权重矩阵上约束方法，alpha_constraint", CONSTRAINTS) = "None",  # type: ignore
    shared_axes: I.str("共享轴，沿着这些轴共享可学习的参数，例如 [1, 2] 表示在高度和宽度轴上共享参数", specific_type_name="list") = None,  # type: ignore
    user_params: I.code(USER_PARAMS_DESC, I.code_python, USER_PARAMS, specific_type_name="函数", auto_complete_type="python") = None,  # type: ignore
    debug: I.bool("输出调试信息，输出更多日志用于debug") = False,  # type: ignore
    input_layer: I.port("输入", optional=True) = None,  # type: ignore
) -> [I.port("layer", "data")]:  # type: ignore
    """PReLU 是 Keras 中的参数化修正线性单元激活层。"""

    import keras

    init_params = dict(
        alpha_initializer=_none(alpha_initializer),
        alpha_regularizer=_none(alpha_regularizer),
        alpha_constraint=_none(alpha_constraint),
        shared_axes=eval(shared_axes) if shared_axes else None,
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

    layer = keras.layers.PReLU(**init_params)(**call_params)

    return I.Outputs(data=layer)


def post_run(outputs):
    """后置运行函数"""
    return outputs