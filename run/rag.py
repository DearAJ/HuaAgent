import os
import json

from dotenv import load_dotenv
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import Chroma
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from langchain_ollama import OllamaEmbeddings
from langchain_openai.chat_models.base import BaseChatOpenAI
from langchain_ollama.chat_models import ChatOllama

# Load environment variables from .env
load_dotenv()

# Define the persistent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
# 从run目录回到4_rag目录的db文件夹
rag_dir = os.path.join(os.path.dirname(current_dir), "4_rag")
persistent_directory = os.path.join(rag_dir, "db", "chroma_db_excel")

# Define the embedding model
embeddings = OllamaEmbeddings(
    model="bge-m3:567m",
    base_url="http://localhost:11434"
)

# Load the existing vector store with the embedding function
db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)

# Create a retriever for querying the vector store
# `search_type` specifies the type of search (e.g., similarity)
# `search_kwargs` contains additional arguments for the search (e.g., number of results to return)
retriever = db.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3},
)

# Create a chat model

# llm = BaseChatOpenAI(
#     model='deepseek-chat',
#     openai_api_key='sk-9dd30006370f4caa93ff65d715f17a0e',
#     openai_api_base='https://api.deepseek.com',
# )

llm = ChatOllama(
    model="qwen3:latest",
    base_url="http://localhost:11434"
)

# Contextualize question prompt
# This system prompt helps the AI understand that it should reformulate the question
# based on the chat history to make it a standalone question
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, just "
    "reformulate it if needed and otherwise return it as is."
)

# Create a prompt template for contextualizing questions
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# Create a history-aware retriever
# This uses the LLM to help reformulate the question based on chat history
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

# Answer question prompt
# This system prompt helps the AI understand that it should provide concise answers
# based on the retrieved context and indicates what to do if the answer is unknown
qa_system_prompt = (
    "You are an assistant for question-answering tasks. Use "
    "the following pieces of retrieved context to answer the "
    "question. If you don't know the answer, just say that you "
    "don't know. Use three sentences maximum and keep the answer "
    "concise."
    "\n\n"
    "{context}"
)

# Create a prompt template for answering questions
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# Create a chain to combine documents for question answering
# `create_stuff_documents_chain` feeds all retrieved context into the LLM
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

# Create a retrieval chain that combines the history-aware retriever and the question answering chain
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

def load_qa_data():
    """从qa_data.json加载问答数据"""
    qa_file_path = os.path.join(current_dir, "qa_data.json")
    with open(qa_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data["qa_pairs"]

def process_qa_pairs():
    """处理所有问答对并保存结果"""
    qa_pairs = load_qa_data()
    results = []
    
    # 确保输出目录存在
    output_dir = os.path.join(current_dir, "return_data", "rag_llm")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "qwen3_bge-m3_return.jsonl")
    
    print(f"开始处理 {len(qa_pairs)} 个问答对...")
    
    for i, qa_pair in enumerate(qa_pairs, 1):
        question = qa_pair["question"]
        expected_answer = qa_pair["answer"]
        
        print(f"处理第 {i}/{len(qa_pairs)} 个问题: {question[:50]}...")
        
        try:
            # 使用RAG链处理问题
            result = rag_chain.invoke({"input": question, "chat_history": []})
            
            # 获取检索到的上下文
            retrieved_contexts = []
            if "context" in result:
                for doc in result["context"]:
                    retrieved_contexts.append(doc.page_content)
            
            # 构建结果记录
            result_record = {
                "user_input": question,
                "response": result["answer"],
                "retrieved_contexts": expected_answer  # 使用qa_pairs中的answer作为retrieved_contexts
            }
            
            results.append(result_record)
            
            # 实时写入JSONL文件
            with open(output_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(result_record, ensure_ascii=False) + '\n')
            
            print(f"✓ 已完成第 {i} 个问题")
            
        except Exception as e:
            print(f"✗ 处理第 {i} 个问题时出错: {str(e)}")
            # 即使出错也记录一个空结果
            result_record = {
                "user_input": question,
                "response": f"处理出错: {str(e)}",
                "retrieved_contexts": expected_answer
            }
            with open(output_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(result_record, ensure_ascii=False) + '\n')
    
    print(f"处理完成！结果已保存到: {output_file}")
    return results

def single_turn_chat():
    print("你好，我是医疗问答AI! 请输入你的问题。")
    query = input("你: ")
    # Process the user's query through the retrieval chain
    result = rag_chain.invoke({"input": query, "chat_history": []})
    # Display the AI's response
    print(f"AI: {result['answer']}")

if __name__ == "__main__":
    # 处理qa_data.json中的所有问答对
    process_qa_pairs()
    
    # 如果需要交互式聊天，可以取消下面的注释
    # single_turn_chat()