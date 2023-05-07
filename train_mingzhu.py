import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import sys
from transformers.integrations import TensorBoardCallback
from torch.utils.tensorboard import SummaryWriter
import fire
import torch
import transformers
from datasets import load_dataset
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)
from transformers import AutoTokenizer, AutoModel


class ModifiedTrainer(transformers.Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        r = model(
            input_ids=inputs["input_ids"],
            labels=inputs["labels"],
        )
        if return_outputs:
            return r.loss, r
        return r.loss

    def save_model(self, output_dir=None, _internal_call=False):
        from transformers.trainer import TRAINING_ARGS_NAME

        os.makedirs(output_dir, exist_ok=True)
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))
        saved_params = {
            k: v.to("cpu") for k, v in self.model.named_parameters() if v.requires_grad
        }
        torch.save(saved_params, os.path.join(output_dir, "adapter_model.bin"))



def train(
    # model/data params
    base_model: str,  # the only required argument
    train_path: str = "data/sanguo_qa.json",
    valid_path: str = None,
    output_dir: str = './lora-glm-sanguo',
    # training hyperparams
    batch_size: int = 32,
    micro_batch_size: int = 1,
    num_epochs: int = 1,
    learning_rate: float = 3e-4,
    cutoff_len: int = 512,
    val_set_size: int = 100,
    # lora hyperparams
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    # llm hyperparams
    train_on_inputs: bool = True,  # if False, masks out inputs in loss
    group_by_length: bool = False,  # faster, but produces an odd training loss curve
    resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
):
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(
            f"Training Alpaca-LoRA model with params:\n"
            f"base_model: {base_model}\n"
            f"train_path: {train_path}\n"
            f"valid_path: {valid_path}\n"
            f"output_dir: {output_dir}\n"
            f"batch_size: {batch_size}\n"
            f"micro_batch_size: {micro_batch_size}\n"
            f"num_epochs: {num_epochs}\n"
            f"learning_rate: {learning_rate}\n"
            f"cutoff_len: {cutoff_len}\n"
            f"val_set_size: {val_set_size}\n"
            f"lora_r: {lora_r}\n"
            f"lora_alpha: {lora_alpha}\n"
            f"lora_dropout: {lora_dropout}\n"
            f"train_on_inputs: {train_on_inputs}\n"
            f"group_by_length: {group_by_length}\n"
            f"resume_from_checkpoint: {resume_from_checkpoint or False}\n"
        )
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='THUDM/chatglm-6b'"
    writer = SummaryWriter()
    gradient_accumulation_steps = batch_size // micro_batch_size

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

    def tokenize(prompt, add_special_tokens=False):
        result = tokenizer(
            prompt,
            truncation=True,
            # max_length=cutoff_len,
            padding=False,
            return_tensors=None,
            add_special_tokens=add_special_tokens
        )
        result["labels"] = result["input_ids"].copy()
        return result

    def generate_and_tokenize_prompt(data_point):
        full_prompt = data_point["context"]
        tokenized_full_prompt = tokenize(full_prompt, True)
        tokenized_output = tokenize(data_point["chat"])

        if not train_on_inputs:
            user_prompt_len = len(tokenized_full_prompt["input_ids"])
            tokenized_full_prompt["labels"] = [-100] * user_prompt_len + tokenized_output["labels"] + [tokenizer.eos_token_id]
        else:
            tokenized_full_prompt["labels"] = tokenized_full_prompt["labels"] + \
                                                 tokenized_output["labels"] + [tokenizer.eos_token_id]
        tokenized_full_prompt["input_ids"] = tokenized_full_prompt["input_ids"] + \
                                             tokenized_output["input_ids"] + [tokenizer.eos_token_id]

        return tokenized_full_prompt

    train_data = load_dataset("json", data_files=train_path)["train"].shuffle().map(generate_and_tokenize_prompt)
    val_data = None
    if valid_path:
        val_data = load_dataset("json", data_files=valid_path)["train"].map(generate_and_tokenize_prompt)

    model = AutoModel.from_pretrained(
        base_model,
        load_in_8bit=True,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        device_map=device_map,
    )

    model = prepare_model_for_int8_training(model)
    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        # target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)

    if resume_from_checkpoint:
        # Check the available weights and load them
        checkpoint_name = os.path.join(
            resume_from_checkpoint, "pytorch_model.bin"
        )  # Full checkpoint
        if not os.path.exists(checkpoint_name):
            checkpoint_name = os.path.join(
                resume_from_checkpoint, "adapter_model.bin"
            )  # only LoRA model - LoRA config above has to fit
            resume_from_checkpoint = (
                False  # So the trainer won't try loading its state
            )
        # The two files above have a different name depending on how they were saved, but are actually the same.
        if os.path.exists(checkpoint_name):
            print(f"Restarting from {checkpoint_name}")
            adapters_weights = torch.load(checkpoint_name)
            model = set_peft_model_state_dict(model, adapters_weights)
        else:
            print(f"Checkpoint {checkpoint_name} not found")

    model.print_trainable_parameters()  # Be more transparent about the % of trainable params.

    if not ddp and torch.cuda.device_count() > 1:
        # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
        model.is_parallelizable = True
        model.model_parallel = True

    trainer = ModifiedTrainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            per_device_eval_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            eval_accumulation_steps=1,
            warmup_steps=50,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            fp16=True,
            logging_steps=1,
            optim="adamw_torch",
            evaluation_strategy="steps" if val_set_size > 0 else "no",
            save_strategy="steps",
            eval_steps=50 if val_set_size > 0 else None,
            save_steps=50,
            output_dir=output_dir,
            save_total_limit=3,
            load_best_model_at_end=True if val_set_size > 0 else False,
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=group_by_length,
        ),
        callbacks=[TensorBoardCallback(writer)],
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )
    model.config.use_cache = False

    old_state_dict = model.state_dict
    model.state_dict = (
        lambda self, *_, **__: get_peft_model_state_dict(
            self, old_state_dict()
        )
    ).__get__(model, type(model))

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    writer.close()

    model.save_pretrained(output_dir)


if __name__ == "__main__":
    fire.Fire(train)
