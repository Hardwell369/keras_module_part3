from bigmodule import I

import structlog

# metadata
author = "BigQuant"
category = r"深度学习\Keras\Attention"
friendly_name = "AdditiveAttention - Keras"
doc_url = "https://keras.io/api/layers/attention_layers/additive_attention/"
cacheable = False

logger = structlog.get_logger()

USER_PARAMS_DESC = """自定义参数"""
USER_PARAMS = """def bigquant_run():
    # https://keras.io/api/layers/attention_layers/additive_attention/

    # import keras

    init_params = dict(
        # use_scale=True,
        # dropout=0.0,
        # **kwargs
    )
    call_params = dict(
        # inputs,
        # training=True,
        # mask=None,
        # return_attention_scores=False,
        # use_causal_mask=False,
    )

    return {"init": init_params, "call": call_params}
"""


def run(
    use_scale: I.bool("是否创建一个标量变量来缩放注意力得分。") = True,  # type: ignore
    dropout: I.float("注意力得分的丢弃率，取值范围为0到1。") = 0.0,  # type: ignore
    query_input: I.port("查询张量输入", optional=False) = None,  # type: ignore
    value_input: I.port("值张量输入", optional=False) = None,  # type: ignore
    key_input: I.port("键张量输入", optional=True) = None,  # type: ignore
    query_mask: I.port("查询掩码输入", optional=True) = None,  # type: ignore
    value_mask: I.port("值掩码输入", optional=True) = None,  # type: ignore
    return_attention_scores: I.bool("是否返回注意力得分（经过掩码和softmax处理后）作为附加输出参数。") = False,  # type: ignore
    training: I.bool("是否在训练模式下运行。") = None,  # type: ignore
    use_causal_mask: I.bool("用于解码器自注意力的掩码。阻止位置i关注位置j > i，防止信息从未来向过去传播。") = False,  # type: ignore
    user_params: I.code(USER_PARAMS_DESC, I.code_python, USER_PARAMS, specific_type_name="函数", auto_complete_type="python") = None,  # type: ignore
    debug: I.bool("输出调试信息，输出更多日志用于debug") = False,  # type: ignore
) -> [
    I.port("attention_output", "attention_output"),
    I.port("attention_scores", "attention_scores"),
    ]:  # type: ignore
    """AdditiveAttention 是一种加性注意力层，也称为 Bahdanau-style 注意力。"""

    import keras

    init_params = dict(
        use_scale=use_scale,
        dropout=dropout,
    )
    call_params = dict(
        inputs=[query_input, value_input, key_input],
        mask=[query_mask, value_mask],
        return_attention_scores=return_attention_scores,
        training=training,
        use_causal_mask=use_causal_mask,
    )

    if user_params is not None:
        user_params = user_params()
        if user_params:
            init_params.update(user_params.get("init", {}))
            call_params.update(user_params.get("call", {}))

    if debug:
        logger.info(f"{init_params=}, {call_params=}")

    attention_output, attention_scores = keras.layers.AdditiveAttention(**init_params)(**call_params)

    return I.Outputs(attention_output=attention_output, attention_scores=attention_scores)


def post_run(outputs):
    """后置运行函数"""
    return outputs