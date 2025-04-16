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
    
    for idx, (content, sol) in enumerate(zip(contents, solution)):
        logger.info(f"===== 样本 {idx+1} 的准确率奖励计算 =====")
        try:
            # 修改后的匹配模式，不再要求<answer>必须在开头
            answer_pattern = r"<answer>(.*?)</answer>"
            pred_match = re.search(answer_pattern, content, re.DOTALL)
            
            if not pred_match:
                logger.warning("【准确率错误】未找到<answer>标签")
                rewards.append(0.0)
                continue
            
            # 提取并处理预测和标准答案
            pred_text = pred_match.group(1).strip()
            pred_stocks = set(preprocess_stock_string(pred_text))
            gold_stocks = set(preprocess_stock_string(sol))
            
            # 详细日志记录
            logger.info(f"【原始答案】\n{pred_text}")
            logger.info(f"【处理后预测】{sorted(list(pred_stocks))}")
            logger.info(f"【标准答案】{sorted(list(gold_stocks))}")
            
            # 处理空集情况
            if not pred_stocks:
                logger.warning("【准确率错误】预测股票列表为空")
                rewards.append(0.0)
                continue
                
            if not gold_stocks:
                logger.warning("【准确率错误】标准答案为空")
                rewards.append(0.0)
                continue
            
            # 计算评估指标
            correct_stocks = pred_stocks & gold_stocks
            incorrect_stocks = pred_stocks - gold_stocks
            missed_stocks = gold_stocks - pred_stocks
            
            correct_count = len(correct_stocks)
            precision = correct_count / len(pred_stocks)
            recall = correct_count / len(gold_stocks)
            
            # 计算F1分数
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            # 完全匹配奖励
            exact_match = 1.0 if pred_stocks == gold_stocks else 0.0
            
            # 组合奖励
            reward = 0.4 * f1 + 0.6 * exact_match
            
            # 详细评估结果
            logger.info("【准确率评估】")
            logger.info(f"  正确预测: {sorted(list(correct_stocks))}")
            if incorrect_stocks:
                logger.warning(f"  错误预测: {sorted(list(incorrect_stocks))}")
            if missed_stocks:
                logger.warning(f"  漏掉股票: {sorted(list(missed_stocks))}")
            logger.info(f"  精确率: {precision:.4f} ({correct_count}/{len(pred_stocks)})")
            logger.info(f"  召回率: {recall:.4f} ({correct_count}/{len(gold_stocks)})")
            logger.info(f"  F1分数: {f1:.4f}")
            logger.info(f"  完全匹配: {'是' if exact_match == 1.0 else '否'}")
            logger.info(f"  最终分数: {reward:.4f} (0.6*F1 + 0.4*完全匹配)")
            
            rewards.append(reward)
            
        except Exception as e:
            logger.error(f"【准确率计算错误】{str(e)}")
            rewards.append(0.0)

    # 汇总输出
    logger.info("===== 准确率奖励汇总 =====")
    for i, reward in enumerate(rewards):
        logger.info(f"样本 {i+1} 的准确率奖励: {reward:.4f}")

    return rewards


def format_reward(completions, **kwargs):
    """格式规范性奖励函数，检查是否符合正则表达式"""
    pattern = r"\s*<think>.*?</think>\s*<answer>.*?</answer>\s*"
    completion_contents = [completion[0]["content"] for completion in completions]
    rewards = []
    
    for idx, content in enumerate(completion_contents):
        logger.info(f"===== 样本 {idx+1} 的格式奖励计算 =====")
        
        # 使用正则表达式进行完整格式验证
        format_correct = bool(re.search(pattern, content, re.DOTALL | re.MULTILINE))
        
        # 计算奖励
        reward = 1.0 if format_correct else 0.0
        
        # 打印简洁信息
        content_brief = content.strip()
        if len(content_brief) > 400:
            content_brief = content_brief[:200] + " [...省略中间内容...] " + content_brief[-200:]
        
        logger.info(f"  格式评分: {reward:.4f} ({'得分' if reward == 1.0 else '未得分'})")
        logger.info(f"  输出内容摘要: {content_brief}")
        
        rewards.append(reward)
    
    # 汇总输出
    logger.info("===== 格式奖励汇总 =====")
    for i, reward in enumerate(rewards):
        logger.info(f"样本 {i+1} 的格式奖励: {reward:.4f}")

    return rewards


def len_reward(completions, solution, **kwargs):
    """股票数量匹配奖励函数"""
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    
    for idx, (content, sol) in enumerate(zip(contents, solution)):
        logger.info(f"===== 样本 {idx+1} 的长度奖励计算 =====")
        try:
            # 提取答案
            answer_pattern = r"<answer>(.*?)</answer>"
            pred_match = re.search(answer_pattern, content, re.DOTALL)
            
            if not pred_match:
                logger.warning("【长度错误】未找到<answer>标签")
                rewards.append(0.0)
                continue
            
            # 处理股票列表
            pred_text = pred_match.group(1).strip()
            pred_stocks = preprocess_stock_string(pred_text)
            gold_stocks = preprocess_stock_string(sol)
            
            pred_count = len(pred_stocks)
            gold_count = len(gold_stocks)
            
           #logger.info(f"【长度比较】")
           # logger.info(f"  预测股票数量: {pred_count}")
           # logger.info(f"  标准答案数量: {gold_count}")
            
            # 精确匹配给满分
            if pred_count == gold_count:
                reward = 1.0
                logger.info(f"  长度评分: {reward:.4f} (数量完全匹配)")
            else:
                reward = 0.0  # 不匹配时不给分
                logger.info(f"  长度评分: {reward:.4f} (数量不匹配)")
            
            rewards.append(reward)
            
        except Exception as e:
            logger.error(f"【长度计算错误】{str(e)}")
            rewards.append(0.0)

    # 汇总输出
    logger.info("===== 长度奖励汇总 =====")
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
