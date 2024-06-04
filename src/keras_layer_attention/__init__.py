from bigmodule import I

import structlog

# metadata
author = "BigQuant"
category = r"深度学习\Keras\Attention"
friendly_name = "Attention - Keras"
doc_url = "https://keras.io/api/layers/attention_layers/attention/"
cacheable = False

logger = structlog.get_logger()

SCORE_MODES = ["dot", "concat"]

USER_PARAMS_DESC = """自定义参数"""
USER_PARAMS = """def bigquant_run():
    # https://keras.io/api/layers/attention_layers/attention/

    # import keras

    init_params = dict(
        # use_scale=False,
        # score_mode="dot",
        # dropout=0.0,
        # seed=None,
        # **kwargs
    )
    call_params = dict(
        # inputs,
        # mask=None,
        # return_attention_scores=False,
        # training=True,
        # use_causal_mask=False,
    )

    return {"init": init_params, "call": call_params}
"""


def run(
    use_scale: I.bool("是否创建一个标量变量来缩放注意力分数", specific_type_name="bool") = False,  # type: ignore
    score_mode: I.choice("用于计算注意力分数的函数", SCORE_MODES) = "dot",  # type: ignore
    dropout: I.float("用于注意力分数的丢弃单元的比例", min=0.0, max=1.0) = 0.0,  # type: ignore
    seed: I.int("用于随机种子的整数", min=0) = None,  # type: ignore
    input_query: I.port("查询张量输入", optional=False) = None,  # type: ignore
    input_value: I.port("值张量输入", optional=False) = None,  # type: ignore
    input_key: I.port("键张量输入", optional=True) = None,  # type: ignore
    query_mask: I.port("查询掩码张量", optional=True) = None,  # type: ignore
    value_mask: I.port("值掩码张量", optional=True) = None,  # type: ignore
    return_attention_scores: I.bool("是否返回注意力分数") = False,  # type: ignore
    training: I.bool("是否在训练模式下运行。") = None,  # type: ignore
    use_causal_mask: I.bool("使用因果掩码") = False,  # type: ignore
    user_params: I.code("自定义函数，自定义设置层构建参数和调用参数", language="python", default=None, specific_type_name="函数", auto_complete_type="python") = None,  # type: ignore
    debug: I.bool("输出调试信息，输出更多日志用于debug") = False,  # type: ignore
) -> [
    I.port("attention_output", "attention_output"),
    I.port("attention_scores", "attention_scores"),
    ]:  # type: ignore
    """Attention 层是用于实现点积注意力机制的层，也称为 Luong 风格注意力。"""

    import keras

    init_params = dict(
        use_scale=use_scale,
        score_mode=score_mode,
        dropout=dropout,
        seed=seed,
    )
    call_params = dict(
        inputs=[input_query, input_value, input_key],
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

    attention_outputs, attention_scores = keras.layers.Attention(**init_params)(**call_params)

    return I.Outputs(attention_output=attention_outputs, attention_scores=attention_scores)


def post_run(outputs):
    """后置运行函数"""
    return outputs