# coding=utf-8
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

"""Reward functions for GRPO training."""

import math
import re
import logging
from typing import Callable, List, Set

# 配置日志记录器
logger = logging.getLogger("reward_functions")
handler = logging.StreamHandler()
handler.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - 【%(levelname)s】 - %(message)s")
)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

def preprocess_stock_string(text):
    """处理各种格式的股票列表字符串
    
    处理以下情况:
    1. 带外层引号: '"正海磁材,丰立智能"' 或 "'正海磁材,丰立智能'"
    2. 不带引号: "正海磁材,丰立智能"
    3. 中文逗号分隔: "正海磁材，丰立智能"
    4. 混合逗号、空格情况
    """
    # 去除最外层引号 (处理多层引号情况)
    while (text.startswith('"') and text.endswith('"')) or (text.startswith("'") and text.endswith("'")):
        text = text[1:-1]
    
    # 替换所有中文逗号，分割并清理每个股票名称
    stocks = []
    for stock in text.replace('，', ',').split(','):
        stock = stock.strip()
        if stock:  # 排除空字符串
            stocks.append(stock)
    
    return stocks


def accuracy_reward(completions, solution, **kwargs):
    """股票预测准确度奖励函数"""
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    
    # 打印完整内容
    logger.info("===== 开始准确率奖励计算 =====")
    for idx, content in enumerate(contents):
        logger.info(f"样本 {idx+1} 的完整内容: {content}")
        # 直接打印 solution[idx] 的内容
        logger.info(f"【标准答案】{solution[idx]}")

    for idx, (content, sol) in enumerate(zip(contents, solution)):
        try:
            answer_pattern = r"<answer>(.*?)</answer>"
            pred_match = re.search(answer_pattern, content, re.DOTALL)
            
            if not pred_match:
                rewards.append(0.0)
                continue
            
            pred_text = pred_match.group(1).strip()
            pred_stocks = set(preprocess_stock_string(pred_text))
            gold_stocks = set(preprocess_stock_string(sol))
            
            if not pred_stocks or not gold_stocks:
                rewards.append(0.0)
                continue
            
            correct_stocks = pred_stocks & gold_stocks
            incorrect_stocks = pred_stocks - gold_stocks
            missed_stocks = gold_stocks - pred_stocks
            
            correct_count = len(correct_stocks)
            # 计算精确率和召回率，避免除以零
            precision = correct_count / len(pred_stocks) if pred_stocks else 0
            recall = correct_count / len(gold_stocks) if gold_stocks else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            exact_match = 1.0 if pred_stocks == gold_stocks else 0.0
            
            # 计算奖励
            reward = 0.7 * f1 + 0.3 * exact_match
            rewards.append(reward)
            
        except Exception as e:
            logger.error(f"样本 {idx+1} 处理时发生异常: {e}")
            rewards.append(0.0)

    # 汇总输出
    logger.info("===== 准确率奖励汇总 =====")
    for i, reward in enumerate(rewards):
        logger.info(f"样本 {i+1} 的准确率奖励: {reward:.4f}")

    return rewards

'''
def format_reward(completions, **kwargs):
    """格式规范性奖励函数，检查是否符合正则表达式"""
    pattern = r"\s*<think>.*?</think>\s*<answer>.*?</answer>\s*"
    completion_contents = [completion[0]["content"] for completion in completions]
    rewards = []
    #logger.info("===== 开始格式奖励计算 =====")
    for idx, content in enumerate(completion_contents):
        format_correct = bool(re.search(pattern, content, re.DOTALL | re.MULTILINE))
        reward = 1.0 if format_correct else 0.0
        rewards.append(reward)
    
    # 汇总输出
    logger.info("===== 格式奖励汇总 =====")
    for i, reward in enumerate(rewards):
        logger.info(f"样本 {i+1} 的格式奖励: {reward:.4f}")

    return rewards
'''

def format_reward(completions, **kwargs):
    """严格格式规范性奖励函数，必须精确包含四个特殊token且符合格式"""
    # 严格匹配模式：允许前后空白，但只能有<think>和<answer>标签各一对
    strict_pattern = re.compile(
        r'^\s*<think>\s*.*?\s*</think>\s*<answer>\s*.*?\s*</answer>\s*$',
        re.DOTALL  # 允许跨行匹配
    )
    
    completion_contents = [completion[0]["content"] for completion in completions]
    rewards = []
    
    for content in completion_contents:
        # 检查1：正则表达式完全匹配整体结构
        format_valid = bool(strict_pattern.fullmatch(content))
        
        # 检查2：确保只有四个特殊token（无重复标签）
        open_tags = re.findall(r'<(think|answer)>', content)
        close_tags = re.findall(r'</(think|answer)>', content)
        tag_count_valid = (
            open_tags == ['think', 'answer'] and  # 标签必须按顺序打开
            close_tags == ['think', 'answer'] and # 标签必须按顺序关闭
            len(open_tags) == 2                   # 只能有两个开标签
        )
        
        # 双重检查均通过才给奖励
        reward = 1.0 if (format_valid and tag_count_valid) else 0.0
        rewards.append(reward)
    
    # 日志输出（假设logger已定义）
    logger.info("===== 严格格式奖励汇总 =====")
    for i, reward in enumerate(rewards):
        logger.info(f"样本 {i+1} 的格式奖励: {reward:.4f}")

    return rewards

def len_reward(completions, solution, **kwargs):
    """股票数量匹配奖励函数"""
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    logger.info("===== 开始股票数目匹配奖励计算 =====")
    for idx, (content, sol) in enumerate(zip(contents, solution)):
        try:
            answer_pattern = r"<answer>(.*?)</answer>"
            pred_match = re.search(answer_pattern, content, re.DOTALL)
            
            if not pred_match:
                rewards.append(0.0)
                continue
            
            pred_text = pred_match.group(1).strip()
            pred_stocks = preprocess_stock_string(pred_text)
            gold_stocks = preprocess_stock_string(sol)
            
            pred_count = len(pred_stocks)
            gold_count = len(gold_stocks)
            
            reward = 1.0 if pred_count == gold_count else 0.0
            rewards.append(reward)
            
        except Exception as e:
            rewards.append(0.0)

    # 汇总输出
    #logger.info("===== 股票数目匹配奖励汇总 =====")
    for i, reward in enumerate(rewards):
        logger.info(f"样本 {i+1} 的长度奖励: {reward:.4f}")

    return rewards


def get_reward_funcs(script_args) -> list[Callable]:
    """获取奖励函数列表"""
    REWARD_FUNCS_REGISTRY = {
        "accuracy": accuracy_reward,
        "format": format_reward,
        "length": len_reward,
    }
    
    # 记录函数加载情况
    logger.info("===== 加载奖励函数 =====")
    logger.info(f"请求加载奖励函数: {script_args.reward_funcs}")
    logger.info(f"对应权重: {script_args.reward_weights}")
    
    # 检查长度是否匹配
    if len(script_args.reward_funcs) != len(script_args.reward_weights):
        logger.error(f"奖励函数数量({len(script_args.reward_funcs)})与权重数量({len(script_args.reward_weights)})不匹配!")
    
    # 检查请求的奖励函数是否都存在
    valid_funcs = []
    valid_weights = []
    
    for i, func_name in enumerate(script_args.reward_funcs):
        if func_name in REWARD_FUNCS_REGISTRY:
            valid_funcs.append(func_name)
            if i < len(script_args.reward_weights):
                valid_weights.append(script_args.reward_weights[i])
            else:
                logger.warning(f"奖励函数 '{func_name}' 缺少对应权重，使用默认值1.0")
                valid_weights.append(1.0)
        else:
            logger.warning(f"奖励函数 '{func_name}' 不存在，可用函数: {list(REWARD_FUNCS_REGISTRY.keys())}")
    
    # 创建带权重的奖励函数
    weighted_reward_funcs = []
    for func_name, weight in zip(valid_funcs, valid_weights):
        base_func = REWARD_FUNCS_REGISTRY[func_name]
        
        # 定义带权重的函数包装器
        def weighted_reward(completions, solution, _func=base_func, _weight=weight, _name=func_name, **kwargs):
            logger.info(f"\n{'='*50}")
            logger.info(f"执行奖励函数: {_name} (权重: {_weight:.4f})")
            raw_rewards = _func(completions, solution, **kwargs)
            weighted_rewards = [r * _weight for r in raw_rewards]
            
            # 记录原始与加权后的奖励
            for i, (raw, weighted) in enumerate(zip(raw_rewards, weighted_rewards)):
                logger.info(f"样本 {i+1}: 原始分数={raw:.4f}, 加权后={weighted:.4f}")
            
            logger.info(f"{'='*50}\n")
            return weighted_rewards
        
        weighted_reward_funcs.append(weighted_reward)
        logger.info(f"已加载奖励函数: {func_name} (权重: {weight:.4f})")
    
    if not weighted_reward_funcs:
        error_msg = f"没有找到有效的奖励函数！请求的函数: {script_args.reward_funcs}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    logger.info(f"成功加载 {len(weighted_reward_funcs)} 个奖励函数")
    return weighted_reward_funcs
