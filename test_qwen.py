from typing import Any, List, Literal, Optional, cast

from evalscope import TaskConfig, run_task
from evalscope.api.messages import ChatMessage
from evalscope.api.model import GenerateConfig, ModelAPI, ModelOutput
from evalscope.api.registry import register_model_api
from evalscope.api.tool import ToolChoice, ToolInfo

from qwen import Qwen3Inference, Qwen3InferenceConfig


@register_model_api(name="qwen3_kv_offload")
class Qwen3KVOffloadModel(ModelAPI):
    def __init__(
        self,
        model_name: str,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        config: GenerateConfig = GenerateConfig(),
        **model_args: Any,
    ) -> None:
        super().__init__(model_name, base_url, api_key, config)
        self.model_args = model_args
        print(self.model_args)

        inference_config = Qwen3InferenceConfig(
            model_path=model_name,
            offload_ratio=float(model_args.get("offload_ratio", 0.6)),
            top_k_per_head=int(model_args.get("top_k_per_head", 8)),
            offload_strategy=cast(
                Literal["middle", "random", "first"],
                model_args.get("offload_strategy", "middle"),
            ),
            temperature=float(model_args.get("temperature", 0.7)),
            top_p=float(model_args.get("top_p", 0.9)),
            top_k=int(model_args.get("top_k", 50)),
            max_seq_len=int(model_args.get("max_seq_len", 1024)),
        )
        self.engine = Qwen3Inference(inference_config)

    def generate(
        self,
        input: List[ChatMessage],
        tools: List[ToolInfo],
        tool_choice: ToolChoice,
        config: GenerateConfig,
    ) -> ModelOutput:
        # 拼接 messages 为单个 prompt 字符串
        prompt = self._process_messages(input)

        max_new_tokens = getattr(config, "max_tokens", None) or 8192

        response = self.engine.generate(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            stream=False,
        )

        return ModelOutput.from_content(model=self.model_name, content=response)

    def _process_messages(self, messages: List[ChatMessage]) -> str:
        parts = []
        for message in messages:
            role = getattr(message, "role", "user")
            content = getattr(message, "content", str(message))
            parts.append(f"{role}: {content}")
        return "\n".join(parts)


def main():
    # 生成参数与原测试保持一致，便于对比
    # 原测试结果参考（main.py，全量 MMLU）：
    #   econometrics:               69.3%  (114 samples)
    #   college_chemistry:          64.0%  (100 samples)
    #   high_school_computer_science: 95.0% (100 samples)
    custom_model = Qwen3KVOffloadModel(
        model_name="/home/wpc/huggingface/Qwen3-8B",
        offload_ratio=0.9,
        top_k_per_head=1,
        offload_strategy="middle",
        temperature=0,
        max_seq_len=8192,
    )

    task_config = TaskConfig(
        model=custom_model,
        datasets=["mmlu"],
        dataset_args={
            "mmlu": {
                "subset_list": [
                    "econometrics",  # 弱项，原始 69.3%，114 题
                    "college_chemistry",  # 最弱 STEM，原始 64.0%，100 题
                    "high_school_computer_science",  # 强项，原始 95.0%，100 题
                ]
            }
        },
    )

    results = run_task(task_cfg=task_config)
    print(results)


if __name__ == "__main__":
    main()
