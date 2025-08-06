import json
import re
import os

def remove_think_tags(text):
    """去除文本中的<think></think>标签及其内容"""
    # 使用正则表达式匹配<think>和</think>之间的所有内容
    # 包括多行内容
    pattern = r'<think>.*?</think>'
    # 使用re.DOTALL标志使.匹配换行符
    cleaned_text = re.sub(pattern, '', text, flags=re.DOTALL)
    return cleaned_text.strip()

def process_jsonl_file(input_file, output_file):
    """处理JSONL文件，去除response字段中的think标签"""
    print(f"开始处理文件: {input_file}")
    
    processed_count = 0
    error_count = 0
    
    try:
        with open(input_file, 'r', encoding='utf-8') as infile, \
             open(output_file, 'w', encoding='utf-8') as outfile:
            
            for line_num, line in enumerate(infile, 1):
                try:
                    # 解析JSON行
                    data = json.loads(line.strip())
                    
                    # 检查是否存在response字段
                    if 'response' in data:
                        original_response = data['response']
                        # 去除think标签
                        cleaned_response = remove_think_tags(original_response)
                        # 更新response字段
                        data['response'] = cleaned_response
                        
                        # 如果response为空，可以设置一个默认值
                        if not cleaned_response:
                            data['response'] = "无答案"
                    
                    # 写入处理后的数据
                    outfile.write(json.dumps(data, ensure_ascii=False) + '\n')
                    processed_count += 1
                    
                    # 每处理100条记录打印一次进度
                    if processed_count % 100 == 0:
                        print(f"已处理 {processed_count} 条记录...")
                        
                except json.JSONDecodeError as e:
                    print(f"第 {line_num} 行JSON解析错误: {e}")
                    error_count += 1
                    continue
                except Exception as e:
                    print(f"第 {line_num} 行处理错误: {e}")
                    error_count += 1
                    continue
    
    except FileNotFoundError:
        print(f"错误: 找不到输入文件 {input_file}")
        return
    except Exception as e:
        print(f"处理文件时出错: {e}")
        return
    
    print(f"\n处理完成!")
    print(f"成功处理: {processed_count} 条记录")
    print(f"错误记录: {error_count} 条")
    print(f"输出文件: {output_file}")

def main():
    # 文件路径
    input_file = "/data/aj/HuaAgent/run/processed_data/qwen3_return.jsonl"
    output_file = "/data/aj/HuaAgent/run/processed_data/qwen3_rmthink_return.jsonl"
    
    # 检查输入文件是否存在
    if not os.path.exists(input_file):
        print(f"错误: 输入文件不存在: {input_file}")
        return
    
    # 处理文件
    process_jsonl_file(input_file, output_file)

if __name__ == "__main__":
    main()
