# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import sys
from dataclasses import dataclass, field

import datasets
import torch
import transformers
from datasets import load_dataset
from transformers import set_seed
from transformers.trainer_utils import get_last_checkpoint

from open_r1.configs import GRPOConfig
from open_r1.rewards import (
    accuracy_reward,
    code_reward,
    format_reward,
    get_code_format_reward,
    get_cosine_scaled_reward,
    get_repetition_penalty_reward,
    len_reward,
    reasoning_steps_reward,
    tag_count_reward,
)
from open_r1.utils import get_tokenizer
from open_r1.utils.callbacks import get_callbacks
from open_r1.utils.wandb_logging import init_wandb_training
from trl import GRPOTrainer, ModelConfig, ScriptArguments, TrlParser, get_peft_config
from swanlab.integration.transformers import SwanLabCallback

logger = logging.getLogger(__name__)

@dataclass
class SwanlabArguments:
    """SwanLab参数的数据类"""
    swanlab: bool = field(
        default=True,
        metadata={"help": "是否使用SwanLab"}
    )
    workspace: str = field(
        default="Lingfeng",
        metadata={"help": "SwanLab工作空间"}
    )
    project: str = field(
        default="CAIYUE",
        metadata={"help": "SwanLab项目名称"}
    )
    experiment_name: str = field(
        default="v1",
        metadata={"help": "SwanLab实验名称"}
    )

# 配置日志记录器
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
)
logger.addHandler(handler)

@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy', 'format', 'reasoning_steps', 'cosine', 'repetition_penalty', 'length', tag_count', 'code', 'code_format'.
        cosine_min_value_wrong (`float`):
            Minimum reward for cosine scaling for wrong answers.
        cosine_max_value_wrong (`float`):
            Maximum reward for cosine scaling for wrong answers.
        cosine_min_value_correct (`float`):
            Minimum reward for cosine scaling for correct answers.
        cosine_max_value_correct (`float`):
            Maximum reward for cosine scaling for correct answers.
        cosine_max_len (`int`):
            Maximum length for cosine scaling.
        code_language (`str`):
            Language for code format reward.
    """#, "tag_count"
    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format","length"],
        metadata={
            #"help": "List of reward functions. Possible values: 'accuracy', 'format', 'reasoning_steps', 'cosine', 'repetition_penalty', 'length', tag_count', 'code', 'code_format'"
            "help": "List of reward functions. Using accuracy and format rewards only."
        },
    )
    cosine_min_value_wrong: float = field(
        default=0.0,
        metadata={"help": "Minimum reward for wrong answers"},
    )
    cosine_max_value_wrong: float = field(
        default=-0.5,
        metadata={"help": "Maximum reward for wrong answers"},
    )
    cosine_min_value_correct: float = field(
        default=0.5,
        metadata={"help": "Minimum reward for correct answers"},
    )
    cosine_max_value_correct: float = field(
        default=1.0,
        metadata={"help": "Maximum reward for correct answers"},
    )
    cosine_max_len: int = field(
        default=1000,
        metadata={"help": "Maximum length for scaling"},
    )
    repetition_n_grams: int = field(
        default=3,
        metadata={"help": "Number of n-grams for repetition penalty reward"},
    )
    repetition_max_penalty: float = field(
        default=-1.0,
        metadata={"help": "Maximum (negative) penalty for for repetition penalty reward"},
    )
    code_language: str = field(
        default="python",
        metadata={
            "help": "Language for code format reward. Based on E2B supported languages https://e2b.dev/docs/code-interpreting/supported-languages",
            "choices": ["python", "javascript", "r", "java", "bash"],
        },
    )
    dataset_name: str = field(
        default="/path/to/your/data",
        metadata={"help": "Dataset name"},
    )
    dataset_train_split: str = field(
        default="train",
        metadata={"help": "Dataset train split"},
    )
    dataset_test_split: str = field(
        default="test",
        metadata={"help": "Dataset test split"},
    )


def main(script_args, training_args, model_args, callbacks):
    # 设置随机种子
    set_seed(training_args.seed)
    
    # 设置日志
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # 检查检查点
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"检测到检查点，从 {last_checkpoint=} 继续训练.")

    if "wandb" in training_args.report_to:
        init_wandb_training(training_args)

    # 加载数据集
    dataset = datasets.DatasetDict({
        'train': load_dataset('json', data_files=os.path.join(script_args.dataset_name, 'train.json'))['train'],
        'test': load_dataset('json', data_files=os.path.join(script_args.dataset_name, 'test.json'))['train']
    })
    
    # 加载分词器
    tokenizer = get_tokenizer(model_args, training_args)

    # 设置奖励函数权重
    REWARD_WEIGHTS = {
        "accuracy": 0.6,
        "format": 0.25,
        "length": 0.15
    }

    # 注册奖励函数
    REWARD_FUNCS_REGISTRY = {
        name: lambda *args, **kwargs: torch.tensor(
            func(*args, **kwargs), 
            device=kwargs.get("device", "cuda")
        ) * REWARD_WEIGHTS[name]
        for name, func in [
            ("accuracy", accuracy_reward),
            ("format", format_reward),
            ("length", len_reward)
        ]
    }

    # 选择奖励函数
    reward_funcs = [REWARD_FUNCS_REGISTRY[func] for func in script_args.reward_funcs]

    # 数据格式化
    def make_conversation(example):
        prompt = []
        if training_args.system_prompt is not None:
            prompt.append({"role": "system", "content": training_args.system_prompt})
        prompt.append({"role": "user", "content": example["problem"]})
        return {
            "prompt": prompt,
            "target": example["output"],
            "solution": example["output"]
        }

    # 处理数据集
    dataset = dataset.map(make_conversation)
    for split in dataset:
        if "messages" in dataset[split].column_names:
            dataset[split] = dataset[split].remove_columns("messages")

    # 初始化模型参数
    logger.info("*** 初始化模型参数 ***")
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )

    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
    )

    # 初始化模型
    model = model_args.model_name_or_path
    if isinstance(model, str):
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            model,
            low_cpu_mem_usage=False,
            **model_kwargs
        )

    # 初始化训练器 - 直接使用实例化后的模型
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        peft_config=get_peft_config(model_args),
        callbacks=get_callbacks(training_args, model_args) + (callbacks if callbacks else [])
    )

    # 训练
    logger.info("*** 开始训练 ***")
    checkpoint = training_args.resume_from_checkpoint or last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    
    # 保存训练结果
    metrics = train_result.metrics
    metrics["train_samples"] = len(dataset[script_args.dataset_train_split])
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    # 保存模型
    logger.info("*** 保存模型 ***")
    trainer.save_model(training_args.output_dir)
    
    # 评估
    if training_args.do_eval:
        logger.info("*** 评估模型 ***")
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(dataset[script_args.dataset_test_split])
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # 推送到hub
    if training_args.push_to_hub:
        trainer.push_to_hub(**{"dataset_name": script_args.dataset_name, "tags": ["open-r1"]})

if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig, SwanlabArguments))
    script_args, training_args, model_args, swanlab_args = parser.parse_args_and_config()
    
    # 初始化SwanLab回调
    if swanlab_args.swanlab:
        swanlab_callback = SwanLabCallback(
            workspace=swanlab_args.workspace,
            project=swanlab_args.project,
            experiment_name=swanlab_args.experiment_name,
        )
        callbacks = [swanlab_callback]
    else:
        callbacks = None
        
    main(script_args, training_args, model_args, callbacks)
