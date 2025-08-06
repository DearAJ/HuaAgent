"""
BleuScore 基于 n-gram 精确率和 简洁性惩罚 来衡量响应与参考文本之间的相似性。
RougeScore 基于 n-gram 召回率、精确率和 F1 分数来衡量生成的 response 与 reference 文本之间的重叠程度。
NonLLMStringSimilarity 指标使用传统的字符串距离度量方法（如 Levenshtein, Hamming, 和 Jaro）来衡量参考文本和响应之间的相似性。

BLEU、ROUGE、NonLLMStringSimilarity 分数范围从0到1，其中1表示响应和参考文本完全匹配。
"""

import json
import asyncio
import os
import glob
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import BleuScore, RougeScore
from ragas.metrics._string import NonLLMStringSimilarity

INPUT_DIR = "/data/aj/HuaAgent/run/processed_data"
OUTPUT_PATH = "/data/aj/HuaAgent/run/evaluate_result/llm_tradition_nlp.json"

async def evaluate_model(jsonl_path):
    """评估单个模型的性能"""
    model_name = os.path.basename(jsonl_path).replace("_return.jsonl", "")
    
    # 读取样本
    samples = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            sample = SingleTurnSample(
                response=data["response"],
                reference=data.get("reference") or data.get("retrieved_contexts") or ""
            )
            samples.append(sample)

    # 计算BLEU分数
    bleu_scorer = BleuScore()
    bleu_scores = []
    for sample in samples:
        try:
            score = await bleu_scorer.single_turn_ascore(sample)
        except Exception as e:
            score = 0.0
        bleu_scores.append(score)
    bleu_avg_score = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0

    # 计算ROUGE分数
    rouge_scorer = RougeScore()
    rouge_scores = []
    for sample in samples:
        try:
            score = await rouge_scorer.single_turn_ascore(sample)
        except Exception as e:
            score = 0.0
        rouge_scores.append(score)
    rouge_avg_score = sum(rouge_scores) / len(rouge_scores) if rouge_scores else 0

    # 计算NonLLMStringSimilarity分数
    string_similarity_scorer = NonLLMStringSimilarity()
    string_similarity_scores = []
    for sample in samples:
        try:
            score = await string_similarity_scorer.single_turn_ascore(sample)
        except Exception as e:
            score = 0.0
        string_similarity_scores.append(score)
    string_similarity_avg_score = sum(string_similarity_scores) / len(string_similarity_scores) if string_similarity_scores else 0
    
    return {
        "model": model_name,
        "bleu_score": bleu_avg_score,
        "rouge_score": rouge_avg_score,
        "string_similarity_score": string_similarity_avg_score
    }

async def main():
    # 获取所有jsonl文件
    jsonl_files = glob.glob(os.path.join(INPUT_DIR, "*_return.jsonl"))
    
    all_results = []
    
    # 遍历每个文件进行评估
    for jsonl_file in jsonl_files:
        print(f"正在评估模型: {os.path.basename(jsonl_file)}")
        result = await evaluate_model(jsonl_file)
        all_results.append(result)
        print(f"完成评估: {result['model']}")
    
    # 写入文件
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    print(f"所有模型评估完成，结果已保存到: {OUTPUT_PATH}")
    
    # 生成折线图
    generate_line_chart(all_results)

def generate_line_chart(results):
    """生成符合科研论文要求的折线图"""
    # 设置科研论文风格的参数
    plt.style.use('default')  # 使用默认样式作为基础
    # 设置字体，优先使用serif字体（在大多数系统上都可用）
    plt.rcParams['font.family'] = 'serif'  # 使用serif字体
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.linewidth'] = 1.0
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['axes.spines.left'] = True
    plt.rcParams['axes.spines.bottom'] = True
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3
    plt.rcParams['grid.linestyle'] = '--'
    plt.rcParams['grid.linewidth'] = 0.5
    
    # 提取数据
    models = [result['model'] for result in results]
    bleu_scores = [result['bleu_score'] for result in results]
    rouge_scores = [result['rouge_score'] for result in results]
    string_similarity_scores = [result['string_similarity_score'] for result in results]
    
    # 创建图表
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 定义科研论文常用的颜色方案
    colors = ['#2E86AB', '#A23B72', '#F18F01']  # 蓝色、紫色、橙色
    markers = ['o', 's', '^']
    labels = ['BLEU Score', 'ROUGE Score', 'String Similarity Score']
    
    # 绘制三条折线
    for i, (scores, color, marker, label) in enumerate(zip([bleu_scores, rouge_scores, string_similarity_scores], colors, markers, labels)):
        ax.plot(models, scores, marker=marker, color=color, linewidth=2.0, markersize=6, 
                label=label, markeredgecolor='white', markeredgewidth=1.0, alpha=0.9)
    
    # 设置图表属性
    ax.set_title('Traditional NLP Metrics Comparison', fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Models', fontsize=12, fontweight='bold')
    ax.set_ylabel('Scores', fontsize=12, fontweight='bold')
    
    # 设置图例
    ax.legend(frameon=True, fancybox=False, shadow=False, fontsize=10, loc='upper right', 
              framealpha=0.9, edgecolor='black', borderpad=0.5)
    
    # 设置网格
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    # 设置y轴范围，从0开始
    ax.set_ylim(bottom=0, top=max(max(bleu_scores), max(rouge_scores), max(string_similarity_scores)) * 1.1)
    
    # 旋转x轴标签
    ax.tick_params(axis='x', rotation=45)
    ax.tick_params(axis='both', which='major', labelsize=10)
    # 设置x轴标签的对齐方式
    plt.setp(ax.get_xticklabels(), ha='right')
    
    # 添加数值标签（只在最高点添加）
    for i, (bleu, rouge, string_sim) in enumerate(zip(bleu_scores, rouge_scores, string_similarity_scores)):
        max_score = max(bleu, rouge, string_sim)
        if max_score > 0.05:  # 只给较高的值添加标签
            ax.annotate(f'{max_score:.3f}', 
                       xy=(i, max_score), 
                       xytext=(0, 5), 
                       textcoords='offset points',
                       ha='center', va='bottom',
                       fontsize=8, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, edgecolor='gray'))
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图表
    chart_path = "/data/aj/HuaAgent/run/evaluate_result/tradition_nlp_chart.png"
    plt.savefig(chart_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"科研级折线图已保存到: {chart_path}")
    
    # 显示图表
    plt.show()

if __name__ == "__main__":
    asyncio.run(main())