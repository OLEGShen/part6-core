import os
import json
import csv
from openai import OpenAI
import time
import traceback
import re

# --- Configuration ---
# 设置您的API密钥和基础URL
# 推荐使用环境变量，如果未设置，请直接替换下面的字符串
API_KEY = os.getenv("DASHSCOPE_API_KEY")
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MODEL_NAME = "qwen-plus"  # 您想使用的模型
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 定义实验文件夹和输出文件
EXPERIMENT_FOLDERS = [f"exp{i}" for i in range(1, 6)]
OUTPUT_CSV_FILE = os.path.join(BASE_DIR, "analysis_results", "llm_thought_analysis.csv")

# 定义LLM分析的提示
ANALYSIS_PROMPT_TEMPLATE = """
Analyze the following thought process of a delivery rider. Your goal is to understand the drivers behind their decisions, paying close attention to factors contributing to intense competition or 'involution'.

Extract keywords in English that represent:
1.  **Rational Aspects:** Focus on logic, efficiency, earnings maximization, route optimization, time management, strategic thinking, order assessment, platform rules, etc.
2.  **Emotional Aspects:** Focus on feelings like stress, pressure, anxiety, fatigue, frustration, satisfaction, urgency, competitiveness, comparison with other riders, desire to outperform, fear of falling behind, etc.

Prioritize keywords that reveal the rider's core motivations and potential reasons for competitive behaviors.

Text:
{text}

Please provide the analysis strictly in the following JSON format, using only English keywords:
{{
  \"rational_keywords\": [\"keyword1\", \"keyword2\", ...],
  \"emotional_keywords\": [\"keyword1\", \"keyword2\", ...]
}}
"""

# --- API Client Setup ---
if not API_KEY:
    raise RuntimeError("请先设置环境变量 DASHSCOPE_API_KEY")

try:
    client = OpenAI(
        api_key=API_KEY,
        base_url=BASE_URL,
    )
except Exception as e:
    print(f"Error initializing OpenAI client: {e}")
    exit(1)

# --- Helper Functions ---
def analyze_thought_with_llm(text):
    """使用LLM分析思考文本。"""
    if not text or not isinstance(text, str) or text.strip() == "":
        print("Skipping analysis for empty or invalid text.")
        return None

    prompt = ANALYSIS_PROMPT_TEMPLATE.format(text=text)
    max_retries = 3
    retry_delay = 5 # seconds

    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": "You are an AI assistant specialized in analyzing thought processes for rationality and emotionality and extracting keywords."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.5, # 控制随机性
            )
            
            response_content = completion.choices[0].message.content.strip() # 去除首尾空白
            print(f"Raw LLM Response: {repr(response_content)}") # 打印原始响应
            
            # 尝试从Markdown代码块中提取JSON
            match = re.search(r"```json\n(.*?)\n```", response_content, re.DOTALL)
            if match:
                json_content = match.group(1).strip()
            else:
                # 如果没有找到代码块，假设整个响应就是JSON（或者至少尝试解析）
                json_content = response_content
                
            # 尝试解析提取出的JSON内容
            try:
                analysis_result = json.loads(json_content)
                # 验证返回的结构
                if isinstance(analysis_result, dict) and \
                   'rational_keywords' in analysis_result and \
                   'emotional_keywords' in analysis_result and \
                   isinstance(analysis_result['rational_keywords'], list) and \
                   isinstance(analysis_result['emotional_keywords'], list):
                    # 解析成功且格式正确
                    pass # 继续到下面的 return analysis_result
                else:
                    # 格式不正确
                    print(f"Warning: LLM response has unexpected format: {analysis_result}")
                    # 尝试提取，如果格式不完全匹配但包含信息
                    rational_kw = analysis_result.get('rational_keywords', ['Format Error'])
                    emotional_kw = analysis_result.get('emotional_keywords', ['Format Error'])
                    if not isinstance(rational_kw, list):
                        rational_kw = [str(rational_kw)] # 保证是列表
                    if not isinstance(emotional_kw, list):
                        emotional_kw = [str(emotional_kw)] # 保证是列表
                    analysis_result = {
                        "rational_keywords": rational_kw,
                        "emotional_keywords": emotional_kw
                    }
                    # 不在此处返回，允许重试循环继续或最终返回错误标记
            except json.JSONDecodeError:
                print(f"Warning: LLM response could not be parsed as JSON after extraction. Extracted content: {repr(json_content)}")
                traceback.print_exc() # 打印堆栈跟踪
                # 如果无法解析JSON，返回错误标记字典
                analysis_result = {
                    "rational_keywords": ["JSON Decode Error"],
                    "emotional_keywords": ["JSON Decode Error"]
                }
                # 不在此处返回，允许重试循环继续或最终返回错误标记

            # 在尝试解析后，再次验证结果格式
            if isinstance(analysis_result, dict) and \
               'rational_keywords' in analysis_result and \
               'emotional_keywords' in analysis_result and \
               isinstance(analysis_result['rational_keywords'], list) and \
               isinstance(analysis_result['emotional_keywords'], list):
                return analysis_result
            else:
                # 处理格式不匹配或之前的解析错误
                print(f"Warning: LLM response has unexpected format or previous parse error: {analysis_result}")
                rational_kw = analysis_result.get('rational_keywords', ['Format/Parse Error'])
                emotional_kw = analysis_result.get('emotional_keywords', ['Format/Parse Error'])
                if not isinstance(rational_kw, list):
                    rational_kw = [str(rational_kw)] # 保证是列表
                if not isinstance(emotional_kw, list):
                    emotional_kw = [str(emotional_kw)] # 保证是列表
                return {
                    "rational_keywords": rational_kw,
                    "emotional_keywords": emotional_kw
                }

        except Exception as e:
            print(f"Error calling LLM API (Attempt {attempt + 1}/{max_retries}):")
            traceback.print_exc() # 打印堆栈跟踪
            if attempt < max_retries - 1:
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                print("Max retries reached. Skipping this analysis.")
                return None
    return None

def process_experiment_folder(folder_path, output_csv_path):
    """处理单个实验文件夹中的所有thought.json文件。"""
    print(f"Processing folder: {folder_path}")
    exp_name = os.path.basename(folder_path)
    for filename in os.listdir(folder_path):
        if filename.endswith("_thought.json"):
            file_path = os.path.join(folder_path, filename)
            try:
                rider_id = filename.split('_')[0]
                with open(file_path, 'r', encoding='utf-8') as f:
                    thoughts_data = json.load(f)
                
                if not isinstance(thoughts_data, list):
                    print(f"Warning: Expected a list in {filename}, got {type(thoughts_data)}. Skipping.")
                    continue

                for thought_entry in thoughts_data:
                    if not isinstance(thought_entry, dict):
                        print(f"Warning: Expected a dict entry in {filename}, got {type(thought_entry)}. Skipping entry.")
                        continue
                        
                    runner_step = thought_entry.get("runner_step", "N/A")
                    think_text = thought_entry.get("think", "")
                    
                    print(f"  Analyzing {exp_name} - Rider {rider_id} - Step {runner_step}...")
                    analysis = analyze_thought_with_llm(think_text)
                    
                    # 实时写入CSV
                    row_data = []
                    if analysis:
                        rational_keywords = json.dumps(analysis.get("rational_keywords", []), ensure_ascii=False)
                        emotional_keywords = json.dumps(analysis.get("emotional_keywords", []), ensure_ascii=False)
                        row_data = [
                            exp_name,
                            rider_id,
                            runner_step,
                            think_text.replace('\n', ' '), # 替换换行符以便CSV存储
                            rational_keywords,
                            emotional_keywords
                        ]
                    else:
                         # 即使分析失败也写入一行，标记为失败
                         row_data = [
                            exp_name,
                            rider_id,
                            runner_step,
                            think_text.replace('\n', ' '), 
                            json.dumps(["Analysis Failed"], ensure_ascii=False),
                            json.dumps(["Analysis Failed"], ensure_ascii=False)
                        ]
                    
                    try:
                        with open(output_csv_path, 'a', newline='', encoding='utf-8') as csvfile_append:
                            csv_writer_append = csv.writer(csvfile_append)
                            csv_writer_append.writerow(row_data)
                    except IOError as e:
                        print(f"Error writing row to CSV {output_csv_path}: {e}")
                    # 添加小的延迟避免触发API速率限制
                    time.sleep(0.5) 

            except FileNotFoundError:
                print(f"Error: File not found {file_path}")
            except json.JSONDecodeError:
                print(f"Error: Could not decode JSON from {file_path}")
            except Exception as e:
                print(f"An unexpected error occurred processing {file_path}:")
                traceback.print_exc() # 打印堆栈跟踪

# --- Main Execution ---
if __name__ == "__main__":
    # 确保输出目录存在
    output_dir = os.path.dirname(OUTPUT_CSV_FILE)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    # 检查CSV文件是否存在，如果不存在则创建并写入表头
    if not os.path.exists(OUTPUT_CSV_FILE):
        try:
            with open(OUTPUT_CSV_FILE, 'w', newline='', encoding='utf-8') as csvfile_init:
                csv_writer_init = csv.writer(csvfile_init)
                # 写入CSV表头
                csv_writer_init.writerow(["Experiment", "RiderID", "Step", "OriginalThinkText", "RationalKeywords", "EmotionalKeywords"])
            print(f"Created CSV file and wrote header: {OUTPUT_CSV_FILE}")
        except IOError as e:
            print(f"Error creating or writing header to CSV file {OUTPUT_CSV_FILE}: {e}")
            exit(1) # 如果无法创建文件，则退出

    try:
        # 获取当前工作目录
        base_dir = BASE_DIR
        
        # 遍历所有实验文件夹 (现在只有一个 'exp1')
        for folder_name in EXPERIMENT_FOLDERS:
            current_folder_path = os.path.join(base_dir, folder_name)
            if os.path.isdir(current_folder_path):
                # 传入CSV文件路径，而不是writer对象
                process_experiment_folder(current_folder_path, OUTPUT_CSV_FILE)
            else:
                print(f"Warning: Folder not found {current_folder_path}")

        print(f"\nAnalysis complete. Results appended to {OUTPUT_CSV_FILE}")

    except Exception as e:
        print(f"An unexpected error occurred during main execution: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during main execution: {e}")
