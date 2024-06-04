from bigmodule import I

import structlog

# metadata
author = "BigQuant"
category = r"深度学习\Keras\Attention"
friendly_name = "GroupedQueryAttention - Keras"
doc_url = "https://keras.io/api/layers/attention_layers/group_query_attention/"
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

REGULARIZERS = [
    "L1",
    "L2",
    "L1L2",
    "OrthogonalRegularizer",
    "Regularizer",
    "None"
]

CONSTRAINTS = [
    "Constraint",
    "MaxNorm",
    "MinMaxNorm",
    "NonNeg",
    "UnitNorm",
    "None"
]

USER_PARAMS_DESC = """自定义参数"""
USER_PARAMS = """def bigquant_run():
    # https://keras.io/api/layers/attention_layers/group_query_attention/

    # import keras

    init_params = dict(
        # head_dim,
        # num_query_heads,
        # num_key_value_heads,
        # dropout=0.0,
        # use_bias=True,
        # kernel_initializer="glorot_uniform",
        # bias_initializer="zeros",
        # kernel_regularizer=None,
        # bias_regularizer=None,
        # activity_regularizer=None,
        # kernel_constraint=None,
        # bias_constraint=None,
        # **kwargs
    )
    call_params = dict(
        # query,
        # value,
        # key=None,
        # attention_mask=None,
        # return_attention_scores=False,
        # training=None,
        # use_causal_mask=False,
    )

    return {"init": init_params, "call": call_params}
"""


def _none(x):
    if x == "None":
        return None
    return x


def run(
    head_dim: I.int("每个注意力头的尺寸。", min=1),  # type: ignore
    num_query_heads: I.int("查询注意力头的数量。", min=1),  # type: ignore
    num_key_value_heads: I.int("键和值注意力头的数量。", min=1),  # type: ignore
    dropout: I.float("dropout概率。", min=0.0, max=1.0) = 0.0,  # type: ignore
    use_bias: I.bool("是否使用偏置项。") = True,  # type: ignore
    kernel_initializer: I.choice("权重矩阵初始化方法，kernel_initializer", INITIALIZERS) = "GlorotUniform",  # type: ignore
    bias_initializer: I.choice("偏置项初始化方法，bias_initializer", INITIALIZERS) = "Zeros",  # type: ignore
    kernel_regularizer: I.choice("权重矩阵正则化方法，kernel_regularizer", REGULARIZERS) = "None",  # type: ignore
    bias_regularizer: I.choice("偏置项正则项，bias_regularizer", REGULARIZERS) = "None",  # type: ignore
    activity_regularizer: I.choice("输出正则项，activity_regularizer", REGULARIZERS) = "None",  # type: ignore
    kernel_constraint: I.choice("权重矩阵上约束方法，kernel_constraint", CONSTRAINTS) = "None",  # type: ignore
    bias_constraint: I.choice("偏置项约束方法，bias_constraint", CONSTRAINTS) = "None",  # type: ignore
    query: I.port("Query tensor", optional=True) = None,  # type: ignore
    value: I.port("Value tensor", optional=True) = None,  # type: ignore
    key: I.port("Key tensor", optional=True) = None,  # type: ignore
    attention_mask: I.port("Attention mask", optional=True) = None,  # type: ignore
    return_attention_scores: I.bool("是否返回注意力分数。") = False,  # type: ignore
    training: I.bool("是否在训练模式下运行。") = None,  # type: ignore
    use_causal_mask: I.bool("是否使用因果蒙版。") = False,  # type: ignore
    user_params: I.code(USER_PARAMS_DESC, I.code_python, USER_PARAMS, specific_type_name="函数", auto_complete_type="python") = None,  # type: ignore
    debug: I.bool("输出调试信息，输出更多日志用于debug") = False,  # type: ignore
) -> [
    I.port("attention_output", "attention_output"),
    I.port("attention_scores", "attention_scores"),
    ]:  # type: ignore
    """GroupedQueryAttention 是 Keras 中用于创建分组查询注意力层的类。"""

    import keras

    init_params = dict(
        head_dim=head_dim,
        num_query_heads=num_query_heads,
        num_key_value_heads=num_key_value_heads,
        dropout=dropout,
        use_bias=use_bias,
        kernel_initializer=_none(kernel_initializer),
        bias_initializer=_none(bias_initializer),
        kernel_regularizer=_none(kernel_regularizer),
        bias_regularizer=_none(bias_regularizer),
        activity_regularizer=_none(activity_regularizer),
        kernel_constraint=_none(kernel_constraint),
        bias_constraint=_none(bias_constraint),
    )
    call_params = dict(
        return_attention_scores=return_attention_scores,
        training=training,
        use_causal_mask=use_causal_mask,
    )
    if query is not None:
        call_params["query"] = query
    if value is not None:
        call_params["value"] = value
    if key is not None:
        call_params["key"] = key
    if attention_mask is not None:
        call_params["attention_mask"] = attention_mask

    if user_params is not None:
        user_params = user_params()
        if user_params:
            init_params.update(user_params.get("init", {}))
            call_params.update(user_params.get("call", {}))

    if debug:
        logger.info(f"{init_params=}, {call_params=}")

    attention_output, attention_scores = keras.layers.GroupQueryAttention(**init_params)(**call_params)

    return I.Outputs(attention_output=attention_output, attention_scores=attention_scores)


def post_run(outputs):
    """后置运行函数"""
    return outputs