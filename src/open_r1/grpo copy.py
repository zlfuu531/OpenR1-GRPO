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
import swanlab


logger = logging.getLogger(__name__)


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
        default_factory=lambda: ["accuracy", "format"],
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


class CustomGRPOTrainer(GRPOTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.device = self.args.device if hasattr(self.args, "device") else "cuda"
        self.model = self.model.to(self.device)

    def compute_rewards(self, prompt_tensors, prompt_mask, completion_tensors, completion_mask):
        # 确保所有输入在同一设备上
        tensors = {
            "prompt_tensors": prompt_tensors.to(self.device),
            "prompt_mask": prompt_mask.to(self.device),
            "completion_tensors": completion_tensors.to(self.device),
            "completion_mask": completion_mask.to(self.device)
        }
        
        # 计算基础奖励
        rewards = super().compute_rewards(**tensors)
        
        # 添加详细日志
        if hasattr(self, 'state') and self.state.global_step > 0:
            logger.info(f"Step {self.state.global_step}")
            logger.info(f"Raw rewards: {rewards}")
            logger.info(f"Reward mean: {rewards.mean()}")
            logger.info(f"Reward std: {rewards.std()}")
            
            # 记录到SwanLab
            if hasattr(self, 'swan') and self.swan is not None:
                self.swan.log({
                    'reward/mean': rewards.mean().item(),
                    'reward/max': rewards.max().item(),
                    'reward/min': rewards.min().item(),
                    'reward/std': rewards.std().item()
                }, step=self.state.global_step)
        
        return rewards

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # 计算基础损失
        loss = super().compute_loss(model, inputs, return_outputs, num_items_in_batch=num_items_in_batch)
        
        # 添加详细日志
        if hasattr(self, 'state') and self.state.global_step > 0:
            logger.info(f"Loss: {loss.item()}")
            logger.info(f"Beta: {self.beta}")
            
            # 记录到SwanLab
            if hasattr(self, 'swan') and self.swan is not None:
                self.swan.log({
                    'training/loss': loss.item(),
                    'training/beta': self.beta
                }, step=self.state.global_step)
        
        return loss


def main(script_args, training_args, model_args):
    # 设置 GPU 设备
    if torch.cuda.is_available():  # 检查是否有可用的GPU
        local_rank = training_args.local_rank  # 获取当前进程的本地 rank
        if local_rank != -1:  # 如果是分布式训练
            # 设置当前进程使用的 GPU 设备
            torch.cuda.set_device(local_rank)
    
    # 设置随机种子，确保实验可重复性
    set_seed(training_args.seed)
    
    ###############
    # 设置日志记录
    ###############
    # 配置日志格式和输出方式
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    # 设置日志级别
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # 在每个进程上输出简要信息摘要
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # 记录模型、脚本和训练参数
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Script parameters {script_args}")
    logger.info(f"Training parameters {training_args}")

    # 检查是否有上一次的检查点
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"检测到检查点，从 {last_checkpoint=} 继续训练.")

    # 如果使用wandb，初始化wandb训练
    if "wandb" in training_args.report_to:
        init_wandb_training(training_args)

    # 初始化 SwanLab 用于实验跟踪和可视化
    swan = None
    if training_args.local_rank in [-1, 0]:  # 只在主进程上初始化 SwanLab
        try:
            swan = swanlab.init(
                api_key="nAkvpigriqq669oTthk4h",
                project_name="GRPO_training",
                display_name=f"GRPO_{model_args.model_name_or_path}",
                resume=True,  # 允许恢复已存在的运行
            )
            
            # 记录配置参数到 SwanLab
            if swan is not None:
                swan.config.update({
                    "model_config": model_args.__dict__,
                    "training_config": training_args.__dict__,
                    "script_config": script_args.__dict__
                })
        except Exception as e:
            logger.warning(f"SwanLab 初始化失败: {str(e)}. 继续训练但不记录到 SwanLab.")
            swan = None

    # 加载数据集
    # 使用DatasetDict来加载训练和测试数据集
    dataset = datasets.DatasetDict({
        'train': load_dataset('json', data_files=os.path.join(script_args.dataset_name, 'train.json'))['train'],
        'test': load_dataset('json', data_files=os.path.join(script_args.dataset_name, 'test.json'))['train']
    })
    
    # 加载分词器
    tokenizer = get_tokenizer(model_args, training_args)

    # 获取奖励函数
    # 设置不同奖励函数的权重
    REWARD_WEIGHTS = {
        "accuracy": 0.6,
        "format": 0.25,
        "length": 0.15
    }

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

    # 根据配置选择使用的奖励函数
    reward_funcs = [REWARD_FUNCS_REGISTRY[func] for func in script_args.reward_funcs]

    # 将数据格式化为对话格式
    def make_conversation(example):
        prompt = []
        # a. 添加系统提示，如果有
        if training_args.system_prompt is not None:
            prompt.append({"role": "system", "content": training_args.system_prompt})
        
        # b. 添加用户提问
        prompt.append({"role": "user", "content": example["problem"]})
        
        # 返回格式化后的数据
        return {
            "prompt": prompt,               # 提示部分（系统提示+用户问题）
            "target": example["output"],    # 目标输出
            "solution": example["output"]   # 解决方案（用于计算奖励）
        }

    # 对数据集应用格式化
    dataset = dataset.map(make_conversation)

    # 如果数据集中有 messages 字段，移除它
    for split in dataset:
        if "messages" in dataset[split].column_names:
            dataset[split] = dataset[split].remove_columns("messages")

    # 初始化模型参数
    logger.info("*** 初始化模型参数 ***")
    # 设置模型的数据类型（如 float16, bfloat16 等）
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    
    # 确定正确的设备
    if torch.cuda.is_available():
        # 在分布式训练中，使用 local_rank 确定设备
        if training_args.local_rank != -1:
            device = f"cuda:{training_args.local_rank}"
        else:
            device = "cuda"
    else:
        device = "cpu"
        
    logger.info(f"Using device: {device}")
    
    # 构建模型初始化参数
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
        logger.info(f"Loading model from {model}")
        model = AutoModelForCausalLM.from_pretrained(
            model,
            low_cpu_mem_usage=False,
            **model_kwargs
        )

    # 确保模型在正确的设备上
    if not training_args.deepspeed or "zero3" not in training_args.deepspeed:
        model = model.to(device)
    model_args.model_name_or_path = model

    # 初始化 GRPO 训练器
    trainer = CustomGRPOTrainer(
        model=model_args.model_name_or_path,  # 模型路径或名称
        reward_funcs=reward_funcs,            # 奖励函数列表
        args=training_args,                   # 训练参数
        train_dataset=dataset[script_args.dataset_train_split],  # 训练数据集
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,  # 评估数据集
        peft_config=get_peft_config(model_args),  # 参数高效微调配置
        callbacks=get_callbacks(training_args, model_args),  # 回调函数
        processing_class=tokenizer,  # 分词器
    )

    # 开始训练
    logger.info("*** 开始训练 ***")
    # 确定是否从检查点继续训练
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    # 执行训练
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    # 获取训练指标
    metrics = train_result.metrics
    metrics["train_samples"] = len(dataset[script_args.dataset_train_split])
    # 记录和保存训练指标
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    # 记录关键指标到 SwanLab
    if swan is not None:
        swan.log({
            'train/loss': metrics['train_loss'],               # 训练损失
            'train/learning_rate': metrics['train_learning_rate'],  # 学习率
            'train/epoch': metrics['epoch'],                   # 当前训练轮次
            'train/global_step': metrics['global_step']        # 全局步数
        })

    # 保存模型和创建模型卡片
    logger.info("*** 保存模型 ***")
    trainer.save_model(training_args.output_dir)
    logger.info(f"模型已保存到 {training_args.output_dir}")

    # 在主进程上保存其他内容
    kwargs = {
        "dataset_name": script_args.dataset_name,
        "tags": ["open-r1"],
    }
    if trainer.accelerator.is_main_process:
        # 创建模型卡片
        trainer.create_model_card(**kwargs)
        # 恢复 k,v cache 以加速推理
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)

    # 评估模型
    if training_args.do_eval:
        logger.info("*** 评估模型 ***")
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(dataset[script_args.dataset_test_split])
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # 推送到hub（如果需要）
    if training_args.push_to_hub:
        logger.info("正在推送到hub...")
        trainer.push_to_hub(**kwargs)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
