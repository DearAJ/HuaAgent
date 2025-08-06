import os
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai.chat_models.base import BaseChatOpenAI
from langchain_ollama.chat_models import ChatOllama

# Load environment variables from .env
load_dotenv()

# Create deepseek-chat model
# llm = BaseChatOpenAI(
#     model='deepseek-chat',
#     openai_api_key='sk-9dd30006370f4caa93ff65d715f17a0e',
#     openai_api_base='https://api.deepseek.com',
# )

llm = ChatOllama(
    model="koesn/llama3-openbiollm-8b:latest",
    base_url="http://localhost:11434"
)

# Answer question prompt (modified for direct LLM response without RAG context)
# This system prompt helps the AI understand that it should provide answers
# based on its general knowledge without specific context
qa_system_prompt = (
    "You are a medical AI assistant for question-answering tasks. "
    "Answer the user's question based on your general medical knowledge. "
    "If you don't know the answer, just say that you don't know. "
    "Use three sentences maximum and keep the answer concise. "
    "Focus on providing accurate and helpful medical information."
)

# Create a prompt template for answering questions
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# Create a simple chain for question answering (without RAG)
def create_simple_qa_chain():
    """Create a simple QA chain without RAG"""
    return qa_prompt | llm

def load_qa_data(file_path: str = "/data/aj/HuaAgent/run/processed_data/qa_data.json"):
    """加载问答数据"""
    print(f"正在加载数据: {file_path}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        qa_pairs = data.get("qa_pairs", [])
        print(f"成功加载 {len(qa_pairs)} 条问答对")
        return qa_pairs
        
    except Exception as e:
        print(f"加载数据时出错: {e}")
        return []

def get_llm_response(question: str) -> str:
    """获取LLM回答"""
    try:
        # Process the user's query through the simple QA chain
        qa_chain = create_simple_qa_chain()
        result = qa_chain.invoke({
            "input": question, 
            "chat_history": []
        })
        
        # Extract the answer from the response
        if hasattr(result, 'content'):
            answer = result.content
        else:
            answer = str(result)
        
        return answer
        
    except Exception as e:
        print(f"获取LLM回答时出错: {e}")
        return f"Error: {e}"

def process_single_question(qa_pair):
    """处理单个问题"""
    question = qa_pair["question"]
    ground_truth_answer = qa_pair["answer"]
    
    # 获取LLM回答
    start_time = time.time()
    llm_response = get_llm_response(question)
    response_time = time.time() - start_time
    
    json_record = {
        "user_input": question,
        "response": llm_response,
        "retrieved_contexts": ground_truth_answer
    }
    
    return {
        "json_record": json_record,
        "response_time": response_time,
        "question": question,
        "llm_response": llm_response
    }

def process_all_questions():
    print("=== 开始处理所有问题（并发版本）===")
    
    # 加载数据
    qa_pairs = load_qa_data()
    if not qa_pairs:
        print("无法加载数据，退出")
        return
    output_file = "/data/aj/HuaAgent/run/processed_data/openbiollm_return.jsonl"
    
    with ThreadPoolExecutor(max_workers=32) as outer_executor:
        futures = []
        
        # 提交所有任务
        for i, qa_pair in enumerate(qa_pairs):
            future = outer_executor.submit(process_single_question, qa_pair)
            futures.append((i, future))
        
        # 收集结果并写入文件
        processed_count = 0
        total_count = len(qa_pairs)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for i, future in futures:
                try:
                    result = future.result()
                    json_record = result["json_record"]
                    response_time = result["response_time"]
                    question = result["question"]
                    llm_response = result["llm_response"]
                    
                    # 写入JSONL文件
                    f.write(json.dumps(json_record, ensure_ascii=False) + '\n')
                    f.flush()  # 确保立即写入文件
                    
                    processed_count += 1
                    print(f"处理问题 {i+1}/{total_count}: {question[:50]}...")
                    print(f"  回答: {llm_response[:100]}...")
                    print(f"  耗时: {response_time:.2f}s")
                    
                except Exception as e:
                    print(f"处理问题 {i+1} 时出错: {e}")
                    # 写入错误记录
                    error_record = {
                        "user_input": qa_pair["question"],
                        "response": f"Error: {e}",
                        "retrieved_contexts": qa_pair["answer"]
                    }
                    f.write(json.dumps(error_record, ensure_ascii=False) + '\n')
                    f.flush()
    
    print(f"\n=== 处理完成 ===")
    print(f"总处理数量: {processed_count}")
    print(f"输出文件: {output_file}")

# Function to simulate a single turn chat
def single_turn_chat():
    print("你好，我是医疗问答AI (Baseline版本)! 请输入你的问题。")
    
    query = input("你: ")
    
    try:
        # Process the user's query through the simple QA chain
        qa_chain = create_simple_qa_chain()
        result = qa_chain.invoke({
            "input": query, 
            "chat_history": []
        })
        
        # Extract the answer from the response
        if hasattr(result, 'content'):
            answer = result.content
        else:
            answer = str(result)
        
        # Display the AI's response
        print(f"AI: {answer}")
        
    except Exception as e:
        print(f"AI: 抱歉，处理您的问题时出现了错误: {e}")
        print("请重新提问或输入'结束'退出对话。")


if __name__ == "__main__":
    process_all_questions()