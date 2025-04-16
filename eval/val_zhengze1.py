import json
import random
import re
from openai import OpenAI
from tqdm import tqdm
import time
import logging
from concurrent.futures import ThreadPoolExecutor

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# 禁用OpenAI和urllib3的HTTP请求日志
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

def load_test_data(json_path):
    """加载测试数据集"""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        logging.error(f"文件未找到: {json_path}")
        exit(1)
    except json.JSONDecodeError:
        logging.error(f"文件格式错误: {json_path}")
        exit(1)
    except Exception as e:
        logging.error(f"加载数据失败: {str(e)}")
        exit(1)

def sample_data(data, sample_size):
    """随机抽样指定数量的测试用例"""
    if sample_size > 0 and len(data) > 0:
        return random.sample(data, min(sample_size, len(data)))
    return data.copy()

def extract_predicted_stocks(response):
    """从模型响应中提取股票名称"""
    try:
        # 尝试匹配<answer>标签
        answer_match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
        content = answer_match.group(1).strip() if answer_match else response
        
        # 匹配中文开头的股票名称（支持带数字）
        stocks = re.findall(r'[\u4e00-\u9fa5][\u4e00-\u9fa5A-Za-z0-9]*', content)
        return [s.strip().upper() for s in stocks if s.strip()]
    except Exception as e:
        logging.error(f"解析失败: {str(e)}")
        return []

def call_model_api(prompt, base_url, model_name, max_retries=3):
    """调用模型API"""
    client = OpenAI(
        api_key="111",
        base_url=base_url,
    )
    
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "system", 
                        "content": "You are a helpful AI Assistant that provides well-reasoned and detailed responses. You first think about the reasoning process as an internal monologue and then provide the user with the answer. Respond in the following format: <think>\n...\n\n<answer>\n...\n</answer>"
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.6,
                top_p=0.95,
                max_tokens=8096,
                timeout=600,
                extra_body={"repetition_penalty": 1.05},
            )
            return response.choices[0].message.content
        except Exception as e:
            logging.warning(f"API调用失败 ({model_name}@{base_url}) 第{attempt+1}次重试: {str(e)}")
            time.sleep(2 ** attempt)  # 指数退避
    logging.error(f"模型 {model_name}@{base_url} 请求失败")
    return ""

def evaluate_single_model(base_url, model_name, sampled_data):
    """评估单个模型"""
    model_results = {
        "base_url": base_url,
        "details": [],
        "statistics": {
            "total_questions": len(sampled_data),
            "total_correct": 0,
            "average_accuracy": 0.0,
            "max_accuracy": 0.0,
            "min_accuracy": 1.0
        }
    }
    
    accuracies = []
    total_correct = 0
    
    for item in tqdm(sampled_data, desc=f"评估 {model_name}", leave=False):
        try:
            # 处理答案
            ground_truth = [s.upper().strip() for s in item['output'].split(',') if s.strip()]
            if not ground_truth:
                logging.warning(f"空答案问题: {item['instruction'][:50]}...")
                continue
            
            # 获取预测
            response = call_model_api(item['instruction'], base_url, model_name)
            predicted = extract_predicted_stocks(response)
            
            # 计算指标
            correct = len(set(ground_truth) & set(predicted))
            accuracy = correct / len(ground_truth) if ground_truth else 0.0
            
            # 记录明细
            model_results["details"].append({
                "ground_truth": ground_truth,
                "predicted": predicted,
                "correct": correct,
                "accuracy": accuracy,
                "raw_response": response
            })
            
            # 更新统计
            total_correct += correct
            accuracies.append(accuracy)
            model_results["statistics"]["max_accuracy"] = max(accuracies) if accuracies else 0.0
            model_results["statistics"]["min_accuracy"] = min(accuracies) if accuracies else 0.0
            
        except Exception as e:
            logging.error(f"[{model_name}] 处理失败: {str(e)}")
            continue

    # 最终统计计算
    if accuracies:
        model_results["statistics"].update({
            "total_correct": total_correct,
            "average_accuracy": sum(accuracies)/len(accuracies),
            "max_accuracy": max(accuracies),
            "min_accuracy": min(accuracies)
        })
    
    return model_results

def evaluate_model(test_data, url_model_map, sample_size=0, output_path="results.json"):
    """执行评估并保存结果（并行版本）"""
    results = {
        "metadata": {
            "sample_size": sample_size if sample_size > 0 else "full",
            "total_questions": len(test_data),
            "models_tested": list(url_model_map.values()),
            "test_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        },
        "results": {}
    }

    # 全局抽样
    sampled_data = sample_data(test_data, sample_size)
    logging.info(f"已抽样 {len(sampled_data)} 个测试用例")

    # 创建线程池（根据CPU核心数调整）
    #max_workers = min(16, len(url_model_map))  # 限制最大并发数
    max_workers = 128
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        
        # 提交所有模型评估任务
        for base_url, model_name in url_model_map.items():
            future = executor.submit(
                evaluate_single_model,
                base_url=base_url,
                model_name=model_name,
                sampled_data=sampled_data
            )
            futures[future] = model_name
            logging.info(f"已提交模型评估任务: {model_name}@{base_url}")

        # 收集结果
        progress_bar = tqdm(futures.items(), desc="总进度", unit="model")
        for future, model_name in progress_bar:
            try:
                model_result = future.result()
                results["results"][model_name] = model_result
                progress_bar.set_postfix_str(f"完成: {model_name}")
            except Exception as e:
                logging.error(f"模型 {model_name} 评估失败: {str(e)}")
                results["results"][model_name] = {"error": str(e)}

    # 保存结果
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logging.info(f"结果已保存至 {output_path}")
    except Exception as e:
        logging.error(f"保存结果失败: {str(e)}")
    
    return results

def print_results(results):
    """打印评估结果"""
    print("\n评估结果汇总:")
    print(f"总问题数: {results['metadata']['total_questions']}")
    print(f"抽样数量: {results['metadata']['sample_size']}")
    print(f"测试时间: {results['metadata']['test_time']}\n")
    
    for model_name, data in results['results'].items():
        if 'error' in data:
            print(f"模型: {model_name} [评估失败]")
            print(f"└── 错误信息: {data['error']}\n")
            continue
        
        stats = data['statistics']
        print(f"模型: {model_name}")
        print(f"├── API端点: {data['base_url']}")
        print(f"├── 评估问题数: {stats['total_questions']}")
        print(f"├── 总正确数: {stats['total_correct']}")
        print(f"├── 平均正确率: {stats['average_accuracy']:.2%}")
        print(f"├── 最高单题正确率: {stats['max_accuracy']:.2%}")
        print(f"└── 最低单题正确率: {stats['min_accuracy']:.2%}\n")

if __name__ == "__main__":
    # 配置参数
    config = {
        "json_path": "/vepfs-d-data/q-caiyue/zlf/data/caiyue48/caiyue48/caiyue48/财联社数据（2025.4.8）/test数据/test.json",
        "url_model_map": {
            "http://0.0.0.0:8018/v1": "LORA_48_V3_20",
            #"http://0.0.0.0:8019/v1": "LORA_48_V3_30",
            "http://0.0.0.0:8020/v1": "LORA_48_V3_40"
            #"http://0.0.0.0:8021/v1": "LORA_48_V3_50",
        },
        "sample_size": 0,
        "output_path": "/vepfs-d-data/q-caiyue/zlf/test_grpo_all3_1_3.json"
    }

    # 执行评估
    start_time = time.time()
    test_data = load_test_data(config["json_path"])
    logging.info(f"成功加载 {len(test_data)} 条测试数据")
    
    results = evaluate_model(
        test_data,
        config["url_model_map"],
        sample_size=config["sample_size"],
        output_path=config["output_path"]
    )

    # 打印结果
    print_results(results)
    logging.info(f"总耗时: {time.time()-start_time:.2f}秒")