import json
import random
import re
from openai import OpenAI
from tqdm import tqdm
import time
import logging
from concurrent.futures import ThreadPoolExecutor
from zhipuai import ZhipuAI

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('evaluation.log'),
        logging.StreamHandler()
    ]
)

# 初始化裁判模型
JUDGE_CLIENT = ZhipuAI(api_key="")  

def load_test_data(json_path):
    """加载测试数据集"""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            logging.info(f"成功加载 {len(data)} 条测试数据")
            return data
    except Exception as e:
        logging.error(f"加载数据失败: {str(e)}")
        raise

def sample_data(data, sample_size):
    """随机抽样"""
    if sample_size > 0 and len(data) > sample_size:
        return random.sample(data, sample_size)
    return data.copy()

def call_model_api(prompt, base_url, model_name, max_retries=3):
    """调用被评估模型API（使用指定参数）"""
    client = OpenAI(
        api_key="empty",
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
                max_tokens=5000,
                timeout=600,
                extra_body={"repetition_penalty": 1.05},
            )
            return response.choices[0].message.content
        except Exception as e:
            logging.warning(f"API调用失败 ({model_name}@{base_url}) 第{attempt+1}次重试: {str(e)}")
            time.sleep(2 ** attempt)
    logging.error(f"模型 {model_name}@{base_url} 请求失败")
    return None

def extract_answer_content(full_response):
    """从模型响应中提取<answer>标签内容"""
    match = re.search(r'<answer>(.*?)</answer>', full_response, re.DOTALL)
    return match.group(1).strip() if match else full_response

def judge_answer(ground_truth, model_response):
    """
    使用GLM-4-Flash评估答案准确率
    返回: 准确率比例 (0.0~1.0)
    """
    # 先提取<answer>内容
    answer_content = extract_answer_content(model_response)
    
    prompt = f"""请严格按照以下规则评估：
1. 提取下列回答中用中文逗号、英文逗号或空格分隔的所有股票名称（不区分大小写和空格）
2. 对比正确答案：[{', '.join(ground_truth)}]
3. 计算：正确识别的股票数量 / 正确答案总数

注意：回答可能包含<think>和<answer>标签，只需评估<answer>内的内容

只输出一个0~1之间的小数，保留4位小数，不要任何其他文字！

示例：
正确答案：A,B,C
回答：<think>思考过程...</think><answer>A,D,C</answer>
输出：0.6667

待评估回答：{answer_content}"""

    for attempt in range(3):
        try:
            response = JUDGE_CLIENT.chat.completions.create(
                model="glm-4-flash",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                top_p=0.1,
                max_tokens=20
            )
            
            # 严格提取数字
            content = response.choices[0].message.content
            match = re.search(r"0?\.\d{1,4}|1\.0{0,4}", content)
            if not match:
                raise ValueError("未找到有效数字")
                
            accuracy = float(match.group())
            return min(max(0.0, accuracy), 1.0)  # 确保在0-1范围内
            
        except Exception as e:
            logging.warning(f"裁判评估失败（第{attempt+1}次）: {str(e)}")
            time.sleep(1)
    
    logging.error(f"无法评估回答：{model_response[:100]}...")
    return 0.0

def evaluate_single_model(base_url, model_name, test_data):
    """评估单个模型"""
    results = {
        "model": model_name,
        "endpoint": base_url,
        "details": [],
        "stats": {
            "total": len(test_data),
            "avg_accuracy": 0.0,
            "score_distribution": {
                "0.0-0.3": 0,
                "0.3-0.6": 0,
                "0.6-0.9": 0,
                "0.9-1.0": 0
            }
        }
    }
    
    accuracies = []
    
    for item in tqdm(test_data, desc=f"评估 {model_name}"):
        try:
            # 准备数据
            question = item['instruction']
            ground_truth = [s.strip().upper() for s in item['output'].split(',') if s.strip()]
            if not ground_truth:
                logging.warning(f"忽略空答案问题: {question[:50]}...")
                continue
            
            # 获取模型回答（使用指定参数）
            response = call_model_api(question, base_url, model_name)
            if not response:
                results["details"].append({
                    "question": question,
                    "error": "获取回答失败"
                })
                continue
                
            # 评估准确率
            accuracy = judge_answer(ground_truth, response)
            accuracies.append(accuracy)
            
            # 记录详情
            results["details"].append({
                "question": question,
                "ground_truth": ground_truth,
                "full_response": response,
                "answer_content": extract_answer_content(response),
                "accuracy": accuracy
            })
            
            # 更新分数分布
            if accuracy <= 0.3:
                results["stats"]["score_distribution"]["0.0-0.3"] += 1
            elif accuracy <= 0.6:
                results["stats"]["score_distribution"]["0.3-0.6"] += 1
            elif accuracy <= 0.9:
                results["stats"]["score_distribution"]["0.6-0.9"] += 1
            else:
                results["stats"]["score_distribution"]["0.9-1.0"] += 1
                
        except Exception as e:
            logging.error(f"处理失败: {str(e)}")
            results["details"].append({
                "question": item.get('instruction', ''),
                "error": str(e)
            })

    # 计算统计指标
    if accuracies:
        results["stats"]["avg_accuracy"] = sum(accuracies) / len(accuracies)
    
    return results

def run_evaluation(config):
    """执行完整评估流程"""
    # 加载数据
    try:
        test_data = load_test_data(config["test_data_path"])
        sampled_data = sample_data(test_data, config.get("sample_size", 0))
        logging.info(f"实际评估数据量: {len(sampled_data)}")
    except Exception as e:
        logging.error(f"初始化失败: {str(e)}")
        return None
    
    # 并行评估
    results = {
        "config": {
            "model_params": {
                "temperature": 0.6,
                "top_p": 0.95,
                "max_tokens": 5000,
                "repetition_penalty": 1.05
            },
            "sample_size": config.get("sample_size", 0)
        },
        "models": [],
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    with ThreadPoolExecutor(max_workers=min(4, len(config["models"]))) as executor:
        futures = []
        for model_cfg in config["models"]:
            future = executor.submit(
                evaluate_single_model,
                base_url=model_cfg["base_url"],
                model_name=model_cfg["name"],
                test_data=sampled_data
            )
            futures.append(future)
        
        for future in tqdm(futures, desc="模型评估进度"):
            try:
                model_result = future.result()
                results["models"].append(model_result)
            except Exception as e:
                logging.error(f"模型评估异常: {str(e)}")
    
    # 保存结果
    try:
        with open(config["output_path"], 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logging.info(f"结果已保存至 {config['output_path']}")
    except Exception as e:
        logging.error(f"保存结果失败: {str(e)}")
    
    return results

def print_summary(results):
    """打印评估摘要"""
    if not results or not results.get("models"):
        print("无有效评估结果")
        return
    
    print(f"\n{' 评估结果汇总 ':^40}")
    print(f"{'='*40}")
    print(f"评估时间: {results['timestamp']}")
    print(f"评估模型数: {len(results['models'])}")
    print(f"样本数量: {results['config']['sample_size'] or '全部'}")
    print(f"{'='*40}")
    
    for model in results["models"]:
        stats = model["stats"]
        print(f"\n模型: {model['model']}")
        print(f"API端点: {model['endpoint']}")
        print(f"评估问题数: {stats['total']}")
        print(f"平均准确率: {stats['avg_accuracy']:.2%}")
        print("\n分数分布:")
        for range_, count in stats["score_distribution"].items():
            print(f"  {range_:<7}: {count:>3} ({count/stats['total']:.1%})")

if __name__ == "__main__":
    # 配置文件
    config = {
        "test_data_path": "/vepfs-d-data/q-caiyue/zlf/data/caiyue/test.json",
        "sample_size": 0,  # 0表示全部数据
        "output_path": "/vepfs-d-data/q-caiyue/zlf/data/caiyue/evaluation_results.json",
        "models": [
             {
                 "name": "lora_old_v1_60",
                 "base_url": "http://0.0.0.0:8001/v1"
             },
        #     {
        #         "name": "lora_new_v1_62",
        #         "base_url": "http://0.0.0.0:8002/v1"
        #     },
        #     {
        #         "name": "lora_old_v1_50",
        #         "base_url": "http://0.0.0.0:8003/v1"
        #     },
        #     {
        #         "name": "lora_old_v1_60",
        #         "base_url": "http://0.0.0.0:8004/v1"
        #     }
            # 可添加更多模型...
        ]
    }
    
    # 执行评估
    start_time = time.time()
    evaluation_results = run_evaluation(config)
    elapsed = time.time() - start_time
    
    # 输出结果
    if evaluation_results:
        print_summary(evaluation_results)
        logging.info(f"评估完成，总耗时: {elapsed:.2f}秒")
