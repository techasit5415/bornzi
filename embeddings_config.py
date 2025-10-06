"""
Embedding Models Configuration
รวบรวม embedding models ต่างๆ ที่ใช้ได้
"""

from langchain_ollama import OllamaEmbeddings
from typing import Literal

class EmbeddingFactory:
    """Factory class สำหรับสร้าง embedding models"""
    
    # รายการ models ที่รองรับ
    SUPPORTED_MODELS = {
        "gemma2:27b": {
            "name": "Gemma 2 27B",
            "description": "Model ใหญ่ ความแม่นยำสูง แต่ช้ากว่า",
            "size": "27B parameters"
        },
        "gemma2:9b": {
            "name": "Gemma 2 9B", 
            "description": "Model กลาง สมดุลระหว่างความเร็วและความแม่นยำ",
            "size": "9B parameters"
        },
        "gemma2:2b": {
            "name": "Gemma 2 2B",
            "description": "Model เล็ก เร็วมาก เหมาะกับเอกสารทั่วไป",
            "size": "2B parameters"
        },
        "nomic-embed-text": {
            "name": "Nomic Embed Text",
            "description": "Model เฉพาะทาง embedding ประสิทธิภาพสูง",
            "size": "137M parameters"
        },
        "mxbai-embed-large": {
            "name": "MixedBread AI Embed Large",
            "description": "Model embedding คุณภาพสูง รองรับหลายภาษา",
            "size": "335M parameters"
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
        return "nomic-embed-text"  # เร็ว เหมาะกับเอกสารเล็ก
    elif document_size < 50000:
        return "gemma2:9b"  # สมดุล
    else:
        return "gemma2:27b"  # คุณภาพสูง สำหรับเอกสารใหญ่
