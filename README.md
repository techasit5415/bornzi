# 📚 PDF RAG Chatbot with Gemma2:27b

RAG Chatbot ที่สามารถอัพโหลด PDF และตอบคำถามจากเอกสารเท่านั้น ใช้ Gemma2:27b ผ่าน Ollama

## � โครงสร้างโปรเจค

```
bornzi/
├── streamlit_app.py          # 🎯 Main Application (รันไฟล์นี้)
├── embeddings_config.py      # ⚙️ Embedding Models Configuration
├── pdf_processor.py          # 📄 PDF Processing & Vector Store
├── llm_config.py            # 🤖 LLM & QA Chain Setup
├── ui_components.py         # 🎨 Streamlit UI Components
├── requirements.txt         # 📦 Dependencies
├── README.md               # 📖 Documentation
├── CONFIGURATION_GUIDE.md  # 📘 Advanced Configuration
└── vectorstore_cache/      # 💾 Cached Vector Stores
```

## �🚀 การติดตั้งและใช้งาน

### 1. ติดตั้ง Dependencies
```bash
pip install -r requirements.txt
```

### 2. ติดตั้งและรัน Ollama
- ดาวน์โหลด Ollama: https://ollama.ai/download
- ติดตั้งและรัน Ollama
- Pull model Gemma2:27b:
```bash
ollama pull gemma2:27b
```

### 3. รันแอพพลิเคชัน
```bash
streamlit run streamlit_app.py
```

## ✨ ฟีเจอร์

- ✅ **Upload PDF** - รองรับเอกสาร PDF ทุกประเภท
- ✅ **ตอบจากเอกสารเท่านั้น** - ไม่หลอนออกนอกทาง ความแม่นยำสูง
- ✅ **เลือก Embedding Model** - 5 models (Gemma 27B/9B/2B, Nomic, MixedBread)
- ✅ **Dynamic Chunk Size** - ปรับอัตโนมัติตามขนาดเอกสาร
- ✅ **Save/Load Vector Store** - ไม่ต้อง process ซ้ำ ประหยัดเวลา
- ✅ **แหล่งอ้างอิง** - แสดง source documents พร้อมหน้าที่มา
- ✅ **Chat History** - บันทึกประวัติการสนทนา
- ✅ **Modular Code** - แยกไฟล์ชัดเจน ง่ายต่อการ maintain

## 🎯 วิธีใช้งาน

1. Upload ไฟล์ PDF ในแถบด้านซ้าย
2. กดปุ่ม "Process PDF"
3. พิมพ์คำถามเกี่ยวกับเอกสาร
4. รับคำตอบที่มาจากเอกสารเท่านั้น

## ⚙️ การตั้งค่า

### LLM Settings
- **Model**: gemma2:27b (Ollama)
- **Temperature**: 0 (แม่นยำสูงสุด)
- **Context Window**: 16,384 tokens
- **Max Predict**: 3,000 tokens

### Retrieval Settings
- **Search Type**: Similarity
- **Retrieved Documents**: 15
- **Chunk Size**: 800-2000 (dynamic)
- **Chunk Overlap**: 150-400 (dynamic)

### Embedding Models
- Gemma2:27b (default) - ความแม่นยำสูง
- Gemma2:9b - สมดุล
- Gemma2:2b - เร็ว
- Nomic Embed Text - เฉพาะทาง
- MixedBread AI - รองรับหลายภาษา

## 📝 หมายเหตุ

- ต้องติดตั้ง Ollama และ pull model gemma2:27b ก่อนใช้งาน
- Ollama จะต้องรันอยู่ที่ localhost:11434
- แอพจะตอบเฉพาะข้อมูลในเอกสารเท่านั้น หากไม่พบจะบอกว่าไม่มีข้อมูล
