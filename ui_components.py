"""
UI Components for Streamlit App
"""
import streamlit as st
from embeddings_config import EmbeddingFactory

def render_sidebar(vectorstore_dir):
    """แสดง Sidebar สำหรับการตั้งค่าและอัพโหลด PDF"""
    
    st.header("📄 Upload PDF Document")
    
    # เลือก AI Model (ใช้ทั้งระบบ)
    st.markdown("### 🤖 AI Model")
    all_models = EmbeddingFactory.get_all_models()
    model_options = {
        name: f"{info['name']} ({info['size']})"
        for name, info in all_models.items()
    }
    
    selected_model = st.selectbox(
        "เลือก Model:",
        options=list(model_options.keys()),
        format_func=lambda x: model_options[x],
        index=list(model_options.keys()).index(st.session_state.embedding_model),
        help="Model ที่ใช้สำหรับทั้ง Embedding และ LLM"
    )
    
    # อัพเดต session state
    if selected_model != st.session_state.embedding_model:
        st.session_state.embedding_model = selected_model
        if st.session_state.pdf_processed:
            st.warning("⚠️ เปลี่ยน model แล้ว กรุณา Process PDF ใหม่")
            st.session_state.pdf_processed = False
            st.session_state.vectorstore = None
    
    # แสดงข้อมูล model
    model_info = EmbeddingFactory.get_model_info(selected_model)
    st.info(f"ℹ️ {model_info.get('description', '')}")
    
    st.markdown("---")
    
    # Upload file
    uploaded_file = st.file_uploader("เลือกไฟล์ PDF", type=['pdf'])
    
    return uploaded_file, model_info


def render_controls():
    """แสดงปุ่มควบคุมเมื่อ process แล้ว"""
    
    if st.session_state.pdf_processed:
        st.markdown("---")
        st.success(f"📌 ไฟล์ปัจจุบัน: {st.session_state.current_pdf_name}")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Clear Chat"):
                st.session_state.chat_history = []
                st.rerun()
        with col2:
            if st.button("🔄 New PDF"):
                st.session_state.vectorstore = None
                st.session_state.pdf_processed = False
                st.session_state.chat_history = []
                st.session_state.current_pdf_name = None
                st.rerun()


def render_instructions():
    """แสดงคำแนะนำการใช้งาน"""
    
    st.markdown("---")
    st.markdown("### ℹ️ วิธีใช้งาน")
    st.markdown("""
    1. Upload ไฟล์ PDF
    2. กด **Process PDF** (ครั้งแรก) หรือ **โหลด Vector Store** (ถ้ามี)
    3. พิมพ์คำถามในช่อง Chat
    4. รับคำตอบจากเอกสาร
    """)


def display_source_documents(response):
    """แสดง source documents"""
    
    with st.expander(f"📑 ดูแหล่งอ้างอิง ({len(response['source_documents'])} แหล่ง)"):
        for i, doc in enumerate(response['source_documents'], 1):
            st.markdown(f"**แหล่งที่ {i}:** (หน้า {doc.metadata.get('page', 'N/A')})")
            
            content = doc.page_content
            if len(content) > 500:
                st.text_area(
                    f"Content {i}",
                    content,
                    height=150,
                    key=f"doc_{i}_{len(st.session_state.chat_history)}"
                )
            else:
                st.text(content)
            st.markdown("---")
