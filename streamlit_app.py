"""
Main Streamlit Application
PDF RAG Chatbot with Gemma2:27b
"""

import streamlit as st
import os

# Import custom modules
from embeddings_config import EmbeddingFactory
from pdf_processor import process_pdf, load_vectorstore, save_vectorstore
# from llm_config import create_qa_chain, get_answer
from llm_config import create_qa_chain, get_answer
from ui_components import render_sidebar, render_controls, render_instructions, display_source_documents


# ตั้งค่า Page
st.set_page_config(page_title="PDF RAG Chatbot", page_icon="📚", layout="wide")
st.title("📚 PDF RAG Chatbot with Gemma2:27b")
st.markdown("---")

# Initialize session state
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'pdf_processed' not in st.session_state:
    st.session_state.pdf_processed = False
if 'current_pdf_name' not in st.session_state:
    st.session_state.current_pdf_name = None
if 'embedding_model' not in st.session_state:
    st.session_state.embedding_model = "gemma2:27b"  # ใช้ DeepSeek เป็นค่าเริ่มต้น

# Constants
VECTORSTORE_DIR = "vectorstore_cache"
os.makedirs(VECTORSTORE_DIR, exist_ok=True)

# ==================== SIDEBAR ====================
with st.sidebar:
    # Render sidebar components
    uploaded_file, model_info = render_sidebar(VECTORSTORE_DIR)
    
    # ตรวจสอบว่ามี Vector Store เก่าหรือไม่
    if uploaded_file is not None:
        vectorstore_path = os.path.join(VECTORSTORE_DIR, f"{uploaded_file.name}.faiss")
        
        # ปุ่มโหลด Vector Store ที่มีอยู่
        if os.path.exists(vectorstore_path) and not st.session_state.pdf_processed:
            if st.button("⚡ โหลด Vector Store ที่มีอยู่", type="secondary"):
                with st.spinner("กำลังโหลด Vector Store..."):
                    try:
                        st.session_state.vectorstore = load_vectorstore(
                            vectorstore_path,
                            st.session_state.embedding_model
                        )
                        st.session_state.pdf_processed = True
                        st.session_state.chat_history = []
                        st.session_state.current_pdf_name = uploaded_file.name
                        st.success(f"✅ โหลด Vector Store สำเร็จ!")
                        st.info(f"🤖 Model: {model_info['name']}")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ ไม่สามารถโหลดได้: {str(e)}")
    
    # ปุ่ม Process PDF
    if uploaded_file is not None:
        if st.button("🔄 Process PDF", type="primary"):
            with st.spinner("กำลังประมวลผล PDF..."):
                try:
                    # ประมวลผล PDF
                    result = process_pdf(uploaded_file, st.session_state.embedding_model)
                    
                    # เก็บผลลัพธ์
                    st.session_state.vectorstore = result['vectorstore']
                    st.session_state.pdf_processed = True
                    st.session_state.chat_history = []
                    st.session_state.current_pdf_name = uploaded_file.name
                    
                    # บันทึก Vector Store
                    vectorstore_path = os.path.join(VECTORSTORE_DIR, f"{uploaded_file.name}.faiss")
                    save_vectorstore(result['vectorstore'], vectorstore_path)
                    
                    # แสดงข้อมูล
                    st.success(f"✅ ประมวลผล PDF สำเร็จ!")
                    st.info(f"📄 ไฟล์: {uploaded_file.name}")
                    st.info(result['chunk_info'])
                    st.info(f"📊 สถิติ: {result['num_pages']} หน้า | {result['total_chars']:,} ตัวอักษร | {result['num_chunks']} chunks")
                    
                    current_model_info = EmbeddingFactory.get_model_info(st.session_state.embedding_model)
                    st.info(f"🤖 Model: {current_model_info['name']} (ใช้ทั้ง Embedding & LLM)")
                    
                    # แนะนำ model
                    if result['recommended_model'] != st.session_state.embedding_model:
                        rec_info = EmbeddingFactory.get_model_info(result['recommended_model'])
                        st.warning(f"💡 แนะนำ: {rec_info['name']} สำหรับเอกสารขนาดนี้")
                    
                    st.info(f"💾 บันทึกที่: `{vectorstore_path}`")
                    
                except Exception as e:
                    st.error(f"❌ เกิดข้อผิดพลาด: {str(e)}")
    
    # แสดงคำแนะนำและปุ่มควบคุม
    render_instructions()
    render_controls()


# ==================== MAIN CHAT INTERFACE ====================
if not st.session_state.pdf_processed:
    st.info("👈 กรุณา Upload และ Process ไฟล์ PDF ก่อนเริ่มสนทนา")
else:
    # แสดงประวัติการสนทนา
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
    
    # Input สำหรับคำถาม
    if prompt := st.chat_input("พิมพ์คำถามเกี่ยวกับเอกสาร..."):
        # แสดงคำถาม
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        
        # สร้างคำตอบ
        with st.chat_message("assistant"):
            with st.spinner("กำลังคิด..."):
                try:
                    # สร้าง QA Chain โดยใช้ model ที่เลือกใน sidebar
                    qa_chain = create_qa_chain(
                        st.session_state.vectorstore,
                        llm_model=st.session_state.embedding_model  # ใช้ model เดียวกัน
                    )
                    
                    # รับคำตอบ
                    response = get_answer(qa_chain, prompt)
                    answer = response['result']
                    
                    # แสดงคำตอบ
                    st.markdown(answer)
                    
                    # แสดง source documents
                    display_source_documents(response)
                    
                    # บันทึกประวัติ
                    st.session_state.chat_history.append({"role": "assistant", "content": answer})
                    
                except Exception as e:
                    error_msg = f"❌ เกิดข้อผิดพลาด: {str(e)}"
                    st.error(error_msg)
                    st.session_state.chat_history.append({"role": "assistant", "content": error_msg})


# ==================== FOOTER ====================
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>🤖 Powered by Gemma2:27b (Ollama) | 🦜 LangChain | 🎈 Streamlit</p>
</div>
""", unsafe_allow_html=True)
