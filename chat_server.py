import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import sys
import fire
import gradio as gr
import torch
from peft import PeftModel
from transformers import GenerationConfig, AutoTokenizer, AutoModel

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:
    pass


def main(
    load_8bit: bool = False,
    base_model: str = "",
    lora_weights: str = "",
    server_name: str = "0.0.0.0",
    share_gradio: bool = False,
):
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='THUDM/chatglm-6b'"
    assert (
        lora_weights
    ), "Please specify a --lora_weights'"

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True, device_map={'':0})
    if device == "cuda":
        model = AutoModel.from_pretrained(
            base_model,
            load_in_8bit=load_8bit,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            device_map='auto',
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            torch_dtype=torch.float16,
            device_map={'': 0}
        )
    elif device == "mps":
        model = AutoModel.from_pretrained(
            base_model,
            device_map={"": device},
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
    else:
        model = AutoModel.from_pretrained(
            base_model, device_map={"": device}, trust_remote_code=True, low_cpu_mem_usage=True
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map={"": device},
        )

    if not load_8bit:
        model.half()

    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    def evaluate(
        input="",
        temperature=0.1,
        top_p=0.75,
        top_k=40,
        num_beams=4,
        max_new_tokens=128,
        **kwargs,
    ):
        inputs = tokenizer(input, return_tensors="pt", add_special_tokens=True)
        input_ids = inputs["input_ids"].to(device)
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            **kwargs,
        )
        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
            )
        s = generation_output.sequences[0]
        output = tokenizer.decode(s)[len(input):]
        return [[input, output]]

    gr.Interface(
        fn=evaluate,
        inputs=[
            gr.components.Textbox(lines=2, label="Input", placeholder="none"),
            gr.components.Slider(
                minimum=0, maximum=1, value=0.1, label="Temperature"
            ),
            gr.components.Slider(
                minimum=0, maximum=1, value=0.75, label="Top p"
            ),
            gr.components.Slider(
                minimum=0, maximum=100, step=1, value=40, label="Top k"
            ),
            gr.components.Slider(
                minimum=1, maximum=4, step=1, value=4, label="Beams"
            ),
            gr.components.Slider(
                minimum=1, maximum=2000, step=1, value=128, label="Max tokens"
            ),
        ],
        outputs=[gr.Chatbot()],
        title="💬Sanguo-ChatGLM-LoRA",
        description="Sanguo-ChatGLM-LoRA是在[ChatGLM-6B](https://github.com/THUDM/ChatGLM-6B)基础上微调三国对话数据得到的，使用的是LoRA的方式微调，主要参考了[Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca)。 更多信息查看[github-LLM_Custome](https://github.com/wellinxu/LLM_Custome)。",
    ).launch(server_name=server_name, share=share_gradio)


if __name__ == "__main__":
    fire.Fire(main)
