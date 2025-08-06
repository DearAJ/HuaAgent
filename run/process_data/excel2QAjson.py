#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
处理Q&A.xlsx文件的前200行数据
提取"问题"列和"答案"列，转换为JSON格式
"""

import pandas as pd
import json
import os
from typing import List, Dict, Any

class QAExcelProcessor:
    """Q&A Excel文件处理器"""
    
    def __init__(self, excel_file_path: str = None):
        # 如果没有指定文件路径，使用当前脚本所在目录下的Q&A.xlsx
        if excel_file_path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            self.excel_file_path = os.path.join(current_dir, "Q&A.xlsx")
        else:
            self.excel_file_path = excel_file_path
        self.data = None
        
    def load_excel_data(self, max_rows: int = 200):
        """加载Excel数据"""
        print(f"正在加载Excel文件: {self.excel_file_path}")
        
        try:
            # 读取Excel文件的前200行
            df = pd.read_excel(self.excel_file_path, nrows=max_rows)
            print(f"成功加载 {len(df)} 行数据")
            
            # 显示列名
            print(f"Excel文件列名: {list(df.columns)}")
            
            self.data = df
            return df
            
        except Exception as e:
            print(f"加载Excel文件时出错: {e}")
            return None
    
    def extract_qa_pairs(self) -> List[Dict[str, Any]]:
        """提取问答对"""
        if self.data is None:
            print("请先加载Excel数据")
            return []
        
        print("正在提取问答对...")
        
        qa_pairs = []
        
        # 查找问题和答案列
        question_col = None
        answer_col = None
        corrected_answer_col = None
        
        # 尝试不同的列名匹配
        possible_question_cols = ['问题', 'question', 'Question', '问题列', '问题内容']
        possible_answer_cols = ['答案', 'answer', 'Answer', '答案列', '答案内容']
        possible_corrected_answer_cols = ['医生更正答案', 'corrected_answer', 'CorrectedAnswer', '更正答案', '医生答案']
        
        for col in self.data.columns:
            if col in possible_question_cols:
                question_col = col
            elif col in possible_answer_cols:
                answer_col = col
            elif col in possible_corrected_answer_cols:
                corrected_answer_col = col
        
        # 如果没有找到标准列名，使用前两列
        if question_col is None or answer_col is None:
            print("未找到标准的问题/答案列名，使用前两列")
            if len(self.data.columns) >= 2:
                question_col = self.data.columns[0]
                answer_col = self.data.columns[1]
            else:
                print("Excel文件列数不足")
                return []
        
        print(f"使用列: 问题='{question_col}', 答案='{answer_col}'")
        if corrected_answer_col:
            print(f"发现医生更正答案列: '{corrected_answer_col}'")
        
        # 统计使用更正答案的数量
        corrected_count = 0
        
        # 提取问答对
        for index, row in self.data.iterrows():
            question = str(row[question_col]) if pd.notna(row[question_col]) else ""
            answer = str(row[answer_col]) if pd.notna(row[answer_col]) else ""
            
            # 检查是否有医生更正答案
            corrected_answer = ""
            used_corrected = False
            
            if corrected_answer_col and pd.notna(row[corrected_answer_col]):
                corrected_answer = str(row[corrected_answer_col]).strip()
                if corrected_answer:  # 如果更正答案不为空
                    answer = corrected_answer
                    used_corrected = True
                    corrected_count += 1
            
            # 跳过空行
            if not question.strip() and not answer.strip():
                continue
            
            qa_pair = {
                "id": index + 1,
                "question": question.strip(),
                "answer": answer.strip(),
                "metadata": {
                    "source": "Q&A.xlsx",
                    "row_index": index + 1,
                    "question_length": len(question.strip()),
                    "answer_length": len(answer.strip()),
                    "used_corrected_answer": used_corrected
                }
            }
            qa_pairs.append(qa_pair)
        
        print(f"成功提取 {len(qa_pairs)} 个问答对")
        if corrected_answer_col:
            print(f"其中使用了 {corrected_count} 个医生更正答案")
        
        return qa_pairs
    
    def save_to_json(self, qa_pairs: List[Dict[str, Any]], output_file: str = None):
        """保存为JSON文件"""
        # 如果没有指定输出文件路径，使用当前脚本所在目录
        if output_file is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            output_file = os.path.join(current_dir, "qa_data.json")
        
        print(f"正在保存到JSON文件: {output_file}")
        
        output_data = {
            "metadata": {
                "source_file": self.excel_file_path,
                "total_pairs": len(qa_pairs),
                "processed_rows": len(self.data) if self.data is not None else 0,
                "created_at": pd.Timestamp.now().isoformat()
            },
            "qa_pairs": qa_pairs
        }
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
            
            print(f"成功保存到 {output_file}")
            return True
            
        except Exception as e:
            print(f"保存JSON文件时出错: {e}")
            return False
    
    def analyze_data(self, qa_pairs: List[Dict[str, Any]]):
        """分析数据统计信息"""
        print("\n=== 数据分析 ===")
        
        if not qa_pairs:
            print("没有数据可分析")
            return
        
        # 统计信息
        total_questions = len(qa_pairs)
        total_question_chars = sum(len(qa["question"]) for qa in qa_pairs)
        total_answer_chars = sum(len(qa["answer"]) for qa in qa_pairs)
        
        avg_question_length = total_question_chars / total_questions if total_questions > 0 else 0
        avg_answer_length = total_answer_chars / total_questions if total_questions > 0 else 0
        
        print(f"总问答对数: {total_questions}")
        print(f"平均问题长度: {avg_question_length:.1f} 字符")
        print(f"平均答案长度: {avg_answer_length:.1f} 字符")
        
        # 长度分布
        short_questions = len([qa for qa in qa_pairs if len(qa["question"]) < 50])
        medium_questions = len([qa for qa in qa_pairs if 50 <= len(qa["question"]) < 200])
        long_questions = len([qa for qa in qa_pairs if len(qa["question"]) >= 200])
        
        print(f"\n问题长度分布:")
        print(f"  短问题 (<50字符): {short_questions}")
        print(f"  中等问题 (50-200字符): {medium_questions}")
        print(f"  长问题 (>=200字符): {long_questions}")
        
        # 显示前几个示例
        print(f"\n前3个问答对示例:")
        for i, qa in enumerate(qa_pairs[:3], 1):
            print(f"\n示例 {i}:")
            print(f"  问题: {qa['question'][:100]}{'...' if len(qa['question']) > 100 else ''}")
            print(f"  答案: {qa['answer'][:100]}{'...' if len(qa['answer']) > 100 else ''}")
        
        return {
            "total_pairs": total_questions,
            "avg_question_length": avg_question_length,
            "avg_answer_length": avg_answer_length,
            "length_distribution": {
                "short": short_questions,
                "medium": medium_questions,
                "long": long_questions
            }
        }

def main():
    """主函数"""
    print("=== Q&A Excel文件处理器 ===")
    
    # 初始化处理器
    processor = QAExcelProcessor()
    
    try:
        # 1. 加载Excel数据
        df = processor.load_excel_data(max_rows=200)
        
        if df is None:
            print("无法加载Excel文件")
            return
        
        # 2. 提取问答对
        qa_pairs = processor.extract_qa_pairs()
        
        if not qa_pairs:
            print("没有提取到有效的问答对")
            return
        
        # 3. 分析数据
        analysis = processor.analyze_data(qa_pairs)
        
        # 4. 保存为JSON文件
        success = processor.save_to_json(qa_pairs, "qa_data.json")
        
        if success:
            print("\n=== 处理完成 ===")
            print("生成的文件:")
            print("- qa_data.json: 问答对数据")
            
            # 显示文件大小
            if os.path.exists("qa_data.json"):
                file_size = os.path.getsize("qa_data.json")
                print(f"- 文件大小: {file_size} 字节 ({file_size/1024:.1f} KB)")
        
    except Exception as e:
        print(f"处理过程中出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 