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
from typing import Callable

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
    """股票预测准确度奖励函数 - 增强错误处理和日志"""
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    
    for content, sol in zip(contents, solution):
        try:
            # 增强的<answer>标签匹配模式
            answer_pattern = r"<answer>\s*([\s\S]*?)\s*</answer>"
            pred_match = re.search(answer_pattern, content, re.IGNORECASE)
            
            if not pred_match:
                print(f"准确率奖励: 无法匹配<answer>标签")
                rewards.append(0.0)
                continue
            
            # 处理股票字符串
            pred_stocks = set(preprocess_stock_string(pred_match.group(1)))
            gold_stocks = set(preprocess_stock_string(sol))
            
            # 调试信息
            print(f"预测股票: {pred_stocks}")
            print(f"标准答案: {gold_stocks}")
            
            # 处理空集情况
            if not pred_stocks:
                print("准确率奖励: 预测股票列表为空")
                rewards.append(0.0)
                continue
                
            if not gold_stocks:
                print("准确率奖励: 标准答案为空")
                rewards.append(0.0)
                continue
            
            # 计算评估指标
            correct_count = len(pred_stocks & gold_stocks)
            precision = correct_count / len(pred_stocks)
            recall = correct_count / len(gold_stocks)
            
            # 计算F1分数
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            # 完全匹配奖励
            exact_match = 1.0 if pred_stocks == gold_stocks else 0.0
            
            # 组合奖励
            reward = 0.6 * f1 + 0.4 * exact_match
            
            print(f"准确率奖励: 精确率={precision:.4f}, 召回率={recall:.4f}, F1={f1:.4f}, 完全匹配={exact_match}, 总分={reward:.4f}")
            rewards.append(reward)
            
        except Exception as e:
            print(f"准确率奖励计算错误: {e}")
            rewards.append(0.0)
    
    return rewards


def format_reward(completions, solution, **kwargs):
    """格式奖励函数 - 严格检查标签存在和顺序
    
    要求：
    1. 四个标签(<think>, </think>, <answer>, </answer>)必须全部存在
    2. 标签顺序必须正确：<think> -> </think> -> <answer> -> </answer>
    3. 全部满足给1分，否则给0分
    """
    completion_contents = [completion[0]["content"] for completion in completions]
    rewards = []
    
    for content, sol in zip(completion_contents, solution):
        try:
            # 检查所有标签是否存在
            think_open = re.search(r"<think>", content, re.IGNORECASE)
            think_close = re.search(r"</think>", content, re.IGNORECASE)
            answer_open = re.search(r"<answer>", content, re.IGNORECASE)
            answer_close = re.search(r"</answer>", content, re.IGNORECASE)
            
            # 所有标签必须存在
            if not all([think_open, think_close, answer_open, answer_close]):
                missing = []
                if not think_open: missing.append("<think>")
                if not think_close: missing.append("</think>")
                if not answer_open: missing.append("<answer>")
                if not answer_close: missing.append("</answer>")
                print(f"格式奖励: 缺少标签: {', '.join(missing)}")
                rewards.append(0.0)
                continue
            
            # 检查标签顺序
            think_open_pos = think_open.start()
            think_close_pos = think_close.start()
            answer_open_pos = answer_open.start()
            answer_close_pos = answer_close.start()
            
            correct_order = (think_open_pos < think_close_pos < answer_open_pos < answer_close_pos)
            
            # 二元奖励：全部满足给1分，否则给0分
            reward = 1.0 if correct_order else 0.0
            
            if reward == 0.0:
                print(f"格式奖励: 标签顺序错误")
            else:
                print(f"格式奖励: 格式正确")
                
            rewards.append(reward)
            
        except Exception as e:
            print(f"格式奖励计算错误: {e}")
            rewards.append(0.0)
    
    return rewards


def len_reward(completions, solution, **kwargs):
    """股票数量匹配奖励函数 - 增强错误处理"""
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    
    for content, sol in zip(contents, solution):
        try:
            # 增强的<answer>标签匹配模式
            answer_pattern = r"<answer>\s*([\s\S]*?)\s*</answer>"
            pred_match = re.search(answer_pattern, content, re.IGNORECASE)
            
            if not pred_match:
                print(f"长度奖励: 无法匹配<answer>标签")
                rewards.append(0.0)
                continue
            
            # 处理股票字符串
            pred_stocks = preprocess_stock_string(pred_match.group(1))
            gold_stocks = preprocess_stock_string(sol)
            
            pred_count = len(pred_stocks)
            gold_count = len(gold_stocks)
            
            print(f"长度奖励: 预测股票数量={pred_count}, 标准答案数量={gold_count}")
            
            # 精确匹配给满分
            if pred_count == gold_count:
                reward = 1.0
                print(f"长度奖励: 数量完全匹配，得分={reward:.4f}")
            else:
                # 差异越大，惩罚越重，使用指数衰减计算
                difference = abs(pred_count - gold_count)
                # 考虑到标准答案股票数量可能不同，使用相对差异
                relative_diff = difference / max(1, gold_count)
                reward = math.exp(-2 * relative_diff)  # 指数衰减函数
                print(f"长度奖励: 数量差异={difference}, 相对差异={relative_diff:.4f}, 得分={reward:.4f}")
            
            rewards.append(reward)
            
        except Exception as e:
            print(f"长度奖励计算错误: {e}")
            rewards.append(0.0)
    
    return rewards


def get_reward_funcs(script_args) -> list[Callable]:
    """获取奖励函数列表
    
    根据script_args中指定的reward_funcs名称列表，
    从注册表中获取对应的奖励函数并返回列表
    """
    REWARD_FUNCS_REGISTRY = {
        "accuracy": accuracy_reward,
        "format": format_reward,
        "length": len_reward,
    }
    
    # 检查请求的奖励函数是否都存在
    for func_name in script_args.reward_funcs:
        if func_name not in REWARD_FUNCS_REGISTRY:
            print(f"警告: 请求的奖励函数 '{func_name}' 不存在，可用函数: {list(REWARD_FUNCS_REGISTRY.keys())}")
    
    # 只返回存在的奖励函数
    reward_funcs = []
    for func_name in script_args.reward_funcs:
        if func_name in REWARD_FUNCS_REGISTRY:
            reward_funcs.append(REWARD_FUNCS_REGISTRY[func_name])
    
    if not reward_funcs:
        raise ValueError(f"没有找到有效的奖励函数！请求的函数: {script_args.reward_funcs}")
        
    return reward_funcs
