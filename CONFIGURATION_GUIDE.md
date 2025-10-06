# 📚 PDF RAG Chatbot - Configuration Guide

## 🎯 การตั้งค่าสำหรับเอกสารประเภทต่างๆ

### 1. เอกสารหลักสูตร / รายวิชา
**ลักษณะ:** ตาราง รหัสวิชา รายการ
```python
chunk_size = 1500-2000
chunk_overlap = 300-400
search_k = 10
fetch_k = 30
lambda_mult = 0.5  # เน้นความหลากหลาย
temperature = 0    # ความแม่นยำสูง
```

### 2. เอกสารทั่วไป / บทความ
**ลักษณะ:** ข้อความยาว ย่อหน้าต่อเนื่อง
```python
chunk_size = 1000-1500
chunk_overlap = 200-300
search_k = 6-8
fetch_k = 20
lambda_mult = 0.7  # สมดุล
temperature = 0.1
```

### 3. เอกสารเทคนิค / Manual
**ลักษณะ:** ขั้นตอน รายละเอียดเฉพาะ
```python
chunk_size = 800-1000
chunk_overlap = 200
search_k = 8
fetch_k = 20
lambda_mult = 0.8  # เน้นความเกี่ยวข้อง
temperature = 0
```

### 4. เอกสารสั้น / FAQ
**ลักษณะ:** คำถาม-คำตอบสั้นๆ
```python
chunk_size = 500-800
chunk_overlap = 100-150
search_k = 5-6
fetch_k = 15
lambda_mult = 0.8
temperature = 0.1
```

## 🔧 พารามิเตอร์สำคัญ

### Chunk Size & Overlap
- **chunk_size**: ขนาดของแต่ละ chunk (characters)
  - เล็ก (500-800): FAQ, เอกสารสั้น
  - กลาง (1000-1500): เอกสารทั่วไป
  - ใหญ่ (1500-2000): หลักสูตร, ตาราง

- **chunk_overlap**: ส่วนที่ซ้อนทับระหว่าง chunks
  - 15-20% ของ chunk_size
  - เพิ่มเพื่อรักษาบริบทข้ามหน้า

### Retrieval Parameters
- **k**: จำนวน documents ที่เลือกใช้
  - 5-6: เอกสารสั้น, คำถามเฉพาะเจาะจง
  - 8-10: เอกสารกลาง-ใหญ่, รายการ

- **fetch_k**: จำนวน candidates ก่อนเลือก
  - แนะนำ: 2-3 เท่าของ k

- **lambda_mult**: สมดุลความเกี่ยวข้อง vs หลากหลาย
  - 0.0 = หลากหลายสูงสุด
  - 1.0 = เกี่ยวข้องสูงสุด
  - แนะนำ: 0.5-0.7

### LLM Parameters
- **temperature**: ความสร้างสรรค์
  - 0 = แม่นยำสูงสุด (หลักสูตร, ข้อมูล)
  - 0.1-0.3 = สมดุล (ทั่วไป)
  - 0.5+ = สร้างสรรค์ (บทสรุป)

- **num_ctx**: context window
  - แนะนำ: 4096-8192
  - เพิ่มถ้ามี documents เยอะ

## 💡 Tips สำหรับผลลัพธ์ที่ดี

1. **ปรับ Prompt ให้ชัดเจน** - บอก LLM ว่าต้องการอะไร
2. **ใช้ MMR แทน Similarity** - ได้ข้อมูลหลากหลายกว่า
3. **เพิ่ม k สำหรับรายการ** - ถ้าคำถามต้องการหลายคำตอบ
4. **ลด temperature** - สำหรับข้อมูลที่ต้องการความแม่นยำ
5. **ตรวจสอบ source documents** - ดูว่า retrieve ถูกหรือไม่

## 🐛 แก้ไขปัญหา

### ตอบไม่ครบ
- ✅ เพิ่ม k และ fetch_k
- ✅ ลด lambda_mult (เน้นหลากหลาย)
- ✅ เพิ่ม chunk_overlap
- ✅ ปรับ prompt ให้บังคับตอบครบ

### ตอบผิด / หลอน
- ✅ ลด temperature เป็น 0
- ✅ เพิ่มกฎใน prompt
- ✅ เพิ่ม lambda_mult (เน้นความเกี่ยวข้อง)

### ช้า
- ✅ ลด k และ fetch_k
- ✅ ลด chunk_size
- ✅ ใช้ embedding model ที่เล็กกว่า

### ใช้ memory เยอะ
- ✅ ลด num_ctx
- ✅ ลด chunk_size
- ✅ ลด k
