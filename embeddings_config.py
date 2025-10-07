"""
Embedding Models Configuration
รวบรวม embedding models ต่างๆ ที่ใช้ได้
"""

from langchain_ollama import OllamaEmbeddings
from typing import Literal

class EmbeddingFactory:
    """Factory class สำหรับสร้าง embedding models"""
    
    # รายการ models ที่รองรับ (ตรงกับที่ติดตั้งใน Ollama)
    SUPPORTED_MODELS = {
        "deepseek-r1:14b": {
            "name": "DeepSeek R1 14B",
            "description": "Model DeepSeek เน้นความเข้าใจลึก มีเหตุผล (แนะนำ)",
            "size": "14B parameters"
        },
        "gemma2:27b": {
            "name": "Gemma 2 27B",
            "description": "Model ใหญ่ที่สุด ความแม่นยำสูง แต่ช้ากว่า",
            "size": "27B parameters"
        },
        "gemma2:9b": {
            "name": "Gemma 2 9B", 
            "description": "Model กลาง สมดุลระหว่างความเร็วและความแม่นยำ",
            "size": "9B parameters"
        },
        "llama3.1:8b": {
            "name": "Llama 3.1 8B",
            "description": "Model จาก Meta เร็วและมีประสิทธิภาพ",
            "size": "8B parameters"
        },
        "gemma3:12b": {
            "name": "Gemma 3 12B",
            "description": "Model Gemma รุ่นใหม่ ประสิทธิภาพดีขึ้น",
            "size": "12B parameters"
        }
    }
    
    @staticmethod
    def create_embeddings(
        model_name: str = "gemma2:27b",
        base_url: str = "http://localhost:11434"
    ) -> OllamaEmbeddings:
        """
        สร้าง embedding model
        
        Args:
            model_name: ชื่อ model ที่ต้องการใช้
            base_url: URL ของ Ollama server
            
        Returns:
            OllamaEmbeddings instance
        """
        if model_name not in EmbeddingFactory.SUPPORTED_MODELS:
            raise ValueError(
                f"Model '{model_name}' ไม่รองรับ. "
                f"Models ที่ใช้ได้: {list(EmbeddingFactory.SUPPORTED_MODELS.keys())}"
            )
        
        return OllamaEmbeddings(
            model=model_name,
            base_url=base_url
        )
    
    @staticmethod
    def get_model_info(model_name: str) -> dict:
        """ดึงข้อมูลของ model"""
        return EmbeddingFactory.SUPPORTED_MODELS.get(model_name, {})
    
    @staticmethod
    def get_all_models() -> dict:
        """ดึงรายการ models ทั้งหมด"""
        return EmbeddingFactory.SUPPORTED_MODELS


def get_recommended_model(document_size: int) -> str:
    """
    แนะนำ model ตามขนาดเอกสาร
    
    Args:
        document_size: จำนวนตัวอักษรในเอกสาร
        
    Returns:
        ชื่อ model ที่แนะนำ
    """
    if document_size < 10000:
        return "llama3.1:8b"  # เร็ว เหมาะกับเอกสารเล็ก
    elif document_size < 50000:
        return "deepseek-r1:14b"  # สมดุล แนะนำ
    else:
        return "gemma2:27b"  # คุณภาพสูงสุด สำหรับเอกสารใหญ่
