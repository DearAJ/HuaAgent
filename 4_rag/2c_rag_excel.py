import os
import pandas as pd
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# Define the directory containing the xlsx file and the persistent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "books", "智能辅助回答审核.xlsx")
persistent_directory = os.path.join(current_dir, "db", "chroma_db_excel")

# Check if the Chroma vector store already exists
if not os.path.exists(persistent_directory):
    print("Persistent directory does not exist. Initializing vector store...")
    
    # Ensure the xlsx file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"The file {file_path} does not exist. Please check the path."
        )
    
    print("Reading Excel file...")
    try:
        df = pd.read_excel(file_path)
        print(f"Excel file loaded successfully. Shape: {df.shape}")
        
        # 显示列名以便调试
        print(f"Columns in Excel: {df.columns.tolist()}")
        
        # 假设列名为：问题、答案、审核状态、医生更正答案
        # 如果列名不同，请相应调整
        question_col = df.columns[0]  # 第一列：问题
        answer_col = df.columns[1]    # 第二列：答案
        status_col = df.columns[2]    # 第三列：审核状态
        corrected_col = df.columns[3] if len(df.columns) > 3 else None  # 第四列：医生更正答案
        
        print(f"Using columns: {question_col}, {answer_col}, {status_col}, {corrected_col}")
        
    except Exception as e:
        raise Exception(f"Error reading Excel file: {str(e)}")
    
    # Convert DataFrame to Document objects
    documents = []
    
    for index, row in df.iterrows():
        question = str(row[question_col]) if pd.notna(row[question_col]) else ""
        answer = str(row[answer_col]) if pd.notna(row[answer_col]) else ""
        status = row[status_col] if pd.notna(row[status_col]) else 0
        corrected_answer = str(row[corrected_col]) if corrected_col and pd.notna(row[corrected_col]) else ""
        
        # 跳过空的问题或答案
        if not question.strip() or not answer.strip():
            continue
            
        # 根据审核状态选择使用原答案还是更正答案
        final_answer = corrected_answer if (status == 1 and corrected_answer.strip()) else answer
        
        # 格式化为问答对话格式
        qa_content = f"Q: {question.strip()}\nA: {final_answer.strip()}"
        
        # 创建Document对象，添加元数据
        doc = Document(
            page_content=qa_content,
            metadata={
                "source": "智能辅助回答审核.xlsx",
                "row_index": index,
                "question": question.strip(),
                "answer": final_answer.strip(),
                "status": status,
                "has_correction": bool(corrected_answer.strip()) if corrected_answer else False
            }
        )
        documents.append(doc)
    
    print(f"Created {len(documents)} documents from Excel data")
    
    # Split the documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,  # 增加chunk_size，因为问答对话通常较长
        chunk_overlap=50,
        separators=["\n\n", "\n", "。", "!", "?", ";", ",", "、", ""],
        length_function=len  # 直接按字符数计算(中文更准确)
    )
    
    docs = text_splitter.split_documents(documents)
    
    # Display information about the split documents
    print("\n--- Document Chunks Information ---")
    print(f"Number of document chunks: {len(docs)}")
    if docs:
        print(f"Sample chunk:\n{docs[0].page_content}\n")
        print(f"Sample metadata:\n{docs[0].metadata}\n")
    
    # Create embeddings
    print("\n--- Creating embeddings ---")
    embeddings = OllamaEmbeddings(
        model="bge-m3:567m",
        base_url="http://localhost:11434"
    )
    print("\n--- Finished creating embeddings ---")
    
    # Create the vector store and persist it automatically
    print("\n--- Creating vector store ---")
    db = Chroma.from_documents(
        docs, embeddings, persist_directory=persistent_directory
    )
    print("\n--- Finished creating vector store ---")
    
    # 显示一些统计信息
    print(f"\n--- Statistics ---")
    print(f"Total QA pairs processed: {len(documents)}")
    print(f"Total document chunks: {len(docs)}")
    
    # 显示审核状态统计
    reviewed_count = sum(1 for doc in documents if doc.metadata.get('status') == 1)
    corrected_count = sum(1 for doc in documents if doc.metadata.get('has_correction'))
    print(f"Reviewed QA pairs: {reviewed_count}")
    print(f"QA pairs with corrections: {corrected_count}")
    
else:
    print("Vector store already exists. No need to initialize.")