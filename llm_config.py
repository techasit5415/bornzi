"""
LLM and QA Chain Configuration
"""
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate


def create_qa_prompt():
    """สร้าง Prompt Template ที่บังคับให้อ่านและตอบตามเอกสารเท่านั้น"""
    
    template = """คุณเป็น AI ที่ตอบคำถามจากเอกสาร ห้ามใช้ความรู้ภายนอกเด็ดขาด

⚠️ ขั้นตอนการทำงาน (ต้องทำทุกขั้น):

ขั้นที่ 1: อ่านเอกสาร
- อ่านเอกสารอ้างอิงทุกบรรทัดอย่างละเอียด
- ห้ามข้ามส่วนใด
- ห้ามคาดเดาหรือสมมติ

ขั้นที่ 2: ค้นหาข้อมูล
- ค้นหาคำตอบในเอกสารที่อ่าน
- ถ้าถามเรื่อง "ปี X เทอม Y" ต้องหาข้อความที่มี "ปี X" หรือ "ชั้นปีที่ X" ตรงตัว
- เก็บข้อมูลทุกอย่างที่เจอ (รหัส ชื่อ ตัวเลข)

ขั้นที่ 3: ตรวจสอบ
- ตรวจว่าข้อมูลที่หาได้มาจากเอกสารหรือไม่
- ถ้าไม่แน่ใจ ห้ามตอบ บอกว่า "ไม่พบข้อมูล"

ขั้นที่ 4: ตอบคำถาม
- ตอบเฉพาะข้อมูลที่พบในเอกสาร
- คัดลอกข้อความตรงตามเอกสาร
- ถ้าถามรายการต้องแสดงครบทุกรายการ

❌ ห้ามทำ:
- ห้ามใช้ความรู้ที่มีอยู่แล้ว
- ห้ามเดาหรือสันนิษฐาน
- ห้ามสรุปหรือตัดทอน
- ห้ามแต่งเติม

📚 เอกสารอ้างอิง:
{context}

❓ คำถาม: {question}

� คิดทีละขั้น:
1. อ่านเอกสารทุกส่วน
2. ค้นหาข้อมูลที่เกี่ยวข้อง
3. ตรวจสอบว่ามาจากเอกสารจริง
4. ตอบเฉพาะที่พบ

💬 คำตอบ:"""
    
    return PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )


def create_llm(model="gemma2:27b", base_url="http://localhost:11434"):
    """สร้าง LLM instance"""
    
    return OllamaLLM(
        model=model,
        base_url=base_url,
        temperature=0,          # แม่นยำสูงสุด
        num_ctx=32768,          # เพิ่มเป็น 32k เพื่อรองรับข้อมูลเยอะ
        num_predict=4096,       # จำนวน token เพิ่มความยาวคำตอบ
        top_k=1,               # เลือกคำที่แน่ใจสุด
        top_p=0.05,            # ลดลงเพื่อความแม่นยำสูงสุด
        repeat_penalty=1.2      # เพิ่มเพื่อป้องกันซ้ำ
    )


def create_qa_chain(vectorstore, llm_model="gemma2:27b", base_url="http://localhost:11434"):
    """
    สร้าง QA Chain
    
    Args:
        vectorstore: FAISS vectorstore
        llm_model: ชื่อ LLM model
        base_url: Ollama base URL
        
    Returns:
        RetrievalQA chain
    """
    
    prompt = create_qa_prompt()
    llm = create_llm(llm_model, base_url)
    
    # ใช้ similarity search และดึงเยอะมากๆ เนื่องจาก chunk เล็ก
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": 30,  # เพิ่มเป็น 30 เพราะ chunk เล็กลง
        }
    )
    
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True,
        verbose=False
    )


def get_answer(qa_chain, question):
    """รับคำตอบจาก QA Chain"""
    return qa_chain({"query": question})
