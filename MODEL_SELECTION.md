# 🔄 Model Selection - ใช้ Model เดียวทั้งระบบ

## ✅ การเปลี่ยนแปลง

ตอนนี้เมื่อคุณเลือก Model ในหน้าเว็บ (Sidebar) จะใช้ Model เดียวกันทั้งระบบ:

### 1️⃣ **Embedding Model** (ตอน Process PDF)
- แปลง PDF เป็น embeddings
- สร้าง Vector Store

### 2️⃣ **LLM Model** (ตอนตอบคำถาม)
- อ่านและเข้าใจคำถาม
- สร้างคำตอบจากเอกสาร

---

## 🎯 วิธีใช้งาน

### ขั้นตอนที่ 1: เลือก Model
ใน Sidebar → **🤖 AI Model**

เลือกได้ 5 รุ่น:
- `gemma2:27b` - ใหญ่ที่สุด แม่นที่สุด (แนะนำ)
- `gemma2:9b` - ขนาดกลาง เร็วกว่า
- `gemma2:2b` - เล็กที่สุด เร็วมาก
- `nomic-embed-text` - เน้น embedding เฉพาะ
- `mxbai-embed-large` - embedding คุณภาพสูง

### ขั้นตอนที่ 2: Upload PDF
อัพโหลดไฟล์ PDF → กด **🔄 Process PDF**

ระบบจะใช้ Model ที่เลือกสร้าง embeddings

### ขั้นตอนที่ 3: ถามคำถาม
พิมพ์คำถาม → ระบบจะใช้ Model เดียวกันตอบ

---

## 📊 ตัวอย่าง

### เดิม (Model แยก):
```
Embedding: gemma2:27b  ← สร้าง embeddings
LLM: gemma2:27b (hard-coded) ← ตอบคำถาม
```
❌ **ปัญหา**: ถ้าเปลี่ยน embedding เป็น gemma2:2b แต่ LLM ยังเป็น gemma2:27b

### ใหม่ (Model เดียว):
```
เลือก: gemma2:2b
  ↓
Embedding: gemma2:2b ← สร้าง embeddings
LLM: gemma2:2b      ← ตอบคำถาม
```
✅ **ข้อดี**: ใช้ model เดียวกันทั้งระบบ สอดคล้องกัน

---

## 💡 คำแนะนำ

### เอกสารขนาดเล็ก (< 10 หน้า)
→ ใช้ `gemma2:2b` (เร็ว ประหยัด)

### เอกสารขนาดกลาง (10-50 หน้า)
→ ใช้ `gemma2:9b` (สมดุล)

### เอกสารขนาดใหญ่ (> 50 หน้า) หรือต้องการความแม่นยำสูง
→ ใช้ `gemma2:27b` (แม่นที่สุด)

---

## 🔧 Technical Details

### ไฟล์ที่เปลี่ยน:

1. **streamlit_app.py** (บรรทัด 133-136):
```python
qa_chain = create_qa_chain(
    st.session_state.vectorstore,
    llm_model=st.session_state.embedding_model  # ← ส่ง model ไปด้วย
)
```

2. **ui_components.py** (บรรทัด 12-26):
- เปลี่ยนจาก "Embedding Model" → "AI Model"
- เปลี่ยน help text เป็น "ใช้ทั้ง Embedding และ LLM"

3. **llm_config.py** & **llm_config_v2.py**:
- รองรับ parameter `llm_model` อยู่แล้ว
- ไม่ต้องแก้ไข

---

## ⚠️ ข้อควรระวัง

1. **เปลี่ยน Model แล้วต้อง Process PDF ใหม่**
   - Vector Store เก่าใช้ไม่ได้กับ model ใหม่
   - ระบบจะเตือนอัตโนมัติ

2. **Model ต่างกันให้ผลต่างกัน**
   - Model ใหญ่ = แม่นแต่ช้า
   - Model เล็ก = เร็วแต่อาจผิดพลาดบ้าง

3. **ต้องติดตั้ง Model ใน Ollama ก่อน**
   ```bash
   ollama pull gemma2:27b
   ollama pull gemma2:9b
   ollama pull gemma2:2b
   ```

---

## 🚀 ทดสอบ

1. เลือก `gemma2:2b` (เล็กสุด)
2. Process PDF
3. ถามคำถาม
4. ดูว่าตอบเร็วหรือไม่ แต่อาจไม่แม่นเท่า gemma2:27b

ลองเปรียบเทียบ! 🎯
