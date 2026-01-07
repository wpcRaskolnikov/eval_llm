from typing import Any, Dict, List, Optional

from evalscope import TaskConfig, run_task
from evalscope.api.messages import ChatMessage
from evalscope.api.model import GenerateConfig, ModelAPI, ModelOutput
from evalscope.api.registry import register_model_api
from evalscope.api.tool import ToolChoice, ToolInfo
from transformers import AutoModelForCausalLM, AutoTokenizer


# 1. Register the model using register_model_api
@register_model_api(name="my_custom_model")
class MyCustomModel(ModelAPI):
    def __init__(
        self,
        model_name: str,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        config: GenerateConfig = GenerateConfig(),
        **model_args: Dict[str, Any],
    ) -> None:
        super().__init__(model_name, base_url, api_key, config)
        self.model_args = model_args
        print(self.model_args)

        # 2. Initialize your model here
        # For example: load model files, establish connections, etc.
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype="auto", device_map="auto"
        )
        self.model.eval()

    def generate(
        self,
        input: List[ChatMessage],
        tools: List[ToolInfo],
        tool_choice: ToolChoice,
        config: GenerateConfig,
    ) -> ModelOutput:
        # 3. Implement model inference logic

        # 3.1 Process input messages
        input_text = self._process_messages(input)

        # 3.2 Call your model
        response = self._call_model(input_text, config)

        # 3.3 Return standardized output
        return ModelOutput.from_content(model=self.model_name, content=response)

    def _process_messages(self, messages: List[ChatMessage]) -> str:
        """Convert chat messages to text"""
        text_parts = []
        for message in messages:
            role = getattr(message, "role", "user")
            content = getattr(message, "content", str(message))
            text_parts.append({"role": role, "content": content})

        text = self.tokenizer.apply_chat_template(
            text_parts,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        return text

    def _call_model(self, input_text: str, config: GenerateConfig) -> str:
        """Invoke your model for inference"""
        # Implement your model invocation logic here
        # For example: API calls, local model inference, etc.

        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.model.device)
        response_ids = self.model.generate(**inputs, max_new_tokens=32768)[0][
            len(inputs.input_ids[0]) :
        ].tolist()
        response = self.tokenizer.decode(response_ids, skip_special_tokens=True)
        return response


def main():
    # Create a model instance
    custom_model = MyCustomModel(
        model_name="/home/wpc/huggingface/Qwen3-8B", model_args={"test": "test"}
    )

    # Configure the evaluation task
    task_config = TaskConfig(
        model=custom_model,
        datasets=["mmlu"],
        # dataset_args={"mmlu": {"subset_list": ["econometrics"]}},
        # limit=100,
    )

    # Run the evaluation
    results = run_task(task_cfg=task_config)
    print(results)


if __name__ == "__main__":
    main()
