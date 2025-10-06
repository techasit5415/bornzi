"""
PDF Processing and Vector Store Management
"""
import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from embeddings_config import EmbeddingFactory, get_recommended_model


def get_dynamic_chunk_params(total_chars, num_pages):
    """คำนวณ chunk size แบบ dynamic - ใช้ chunk เล็กมากและ overlap สูงมากเพื่อไม่ให้ข้อมูลหาย"""
    
    # ลด chunk size และเพิ่ม overlap เพื่อให้ context ละเอียดและครอบคลุมมากขึ้น
    if total_chars < 10000:
        return {
            'chunk_size': 400,
            'chunk_overlap': 250,  # 62.5% overlap - มากกว่าเดิมเพื่อให้แน่ใจว่าไม่หาย
            'info': f"📄 เอกสารขนาดเล็ก ({num_pages} หน้า) - chunk: 400, overlap: 250 (62.5%)"
        }
    elif total_chars < 50000:
        return {
            'chunk_size': 500,
            'chunk_overlap': 300,  # 60% overlap
            'info': f"📄 เอกสารขนาดกลาง ({num_pages} หน้า) - chunk: 500, overlap: 300 (60%)"
        }
    else:
        return {
            'chunk_size': 600,
            'chunk_overlap': 360,  # 60% overlap
            'info': f"📄 เอกสารขนาดใหญ่ ({num_pages} หน้า) - chunk: 600, overlap: 360 (60%)"
        }


def process_pdf(uploaded_file, embedding_model, base_url="http://localhost:11434"):
    """
    ประมวลผล PDF และสร้าง Vector Store
    
    Returns:
        dict: {
            'vectorstore': FAISS vectorstore,
            'num_pages': จำนวนหน้า,
            'total_chars': จำนวนตัวอักษร,
            'num_chunks': จำนวน chunks,
            'recommended_model': model ที่แนะนำ
        }
    """
    
    # บันทึกไฟล์ชั่วคราว
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name
    
    try:
        # โหลด PDF
        loader = PyPDFLoader(tmp_file_path)
        documents = loader.load()
        
        # คำนวณขนาด
        total_chars = sum(len(doc.page_content) for doc in documents)
        num_pages = len(documents)
        
        # แนะนำ model
        recommended_model = get_recommended_model(total_chars)
        
        # Dynamic chunk size
        chunk_params = get_dynamic_chunk_params(total_chars, num_pages)
        
        # แบ่งข้อความ - ใช้ chunk เล็กและ overlap สูงมากเพื่อความครอบคลุม
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_params['chunk_size'],
            chunk_overlap=chunk_params['chunk_overlap'],
            length_function=len,
            separators=["\n\n", "\n", " ", ""],  # ไม่ใช้ . เพื่อไม่ให้ตัดประโยค
            keep_separator=True  # เก็บ separator เพื่อรักษาโครงสร้าง
        )
        texts = text_splitter.split_documents(documents)
        
        # สร้าง Embeddings
        embeddings = EmbeddingFactory.create_embeddings(
            model_name=embedding_model,
            base_url=base_url
        )
        
        # สร้าง Vector Store
        vectorstore = FAISS.from_documents(texts, embeddings)
        
        return {
            'vectorstore': vectorstore,
            'num_pages': num_pages,
            'total_chars': total_chars,
            'num_chunks': len(texts),
            'recommended_model': recommended_model,
            'chunk_info': chunk_params['info']
        }
        
    finally:
        # ลบไฟล์ชั่วคราว
        os.unlink(tmp_file_path)


def load_vectorstore(vectorstore_path, embedding_model, base_url="http://localhost:11434"):
    """โหลด Vector Store จากไฟล์"""
    
    embeddings = EmbeddingFactory.create_embeddings(
        model_name=embedding_model,
        base_url=base_url
    )
    
    return FAISS.load_local(
        vectorstore_path,
        embeddings,
        allow_dangerous_deserialization=True
    )


def save_vectorstore(vectorstore, vectorstore_path):
    """บันทึก Vector Store ลงไฟล์"""
    vectorstore.save_local(vectorstore_path)
