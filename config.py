import os
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    hf_api_token: str = os.getenv("HF_API_TOKEN", "")
    hf_model_id: str = os.getenv("HF_MODEL_ID", "Qwen/Qwen2.5-72B-Instruct")
    hf_image_model: str = os.getenv("HF_IMAGE_MODEL", "black-forest-labs/FLUX.1-schnell")
    faiss_index_path: str = os.getenv("FAISS_INDEX_PATH", "memory/faiss_index")
    output_dir: str = os.getenv("OUTPUT_DIR", "outputs")
    raw_scenes_dir: str = os.getenv("RAW_SCENES_DIR", "raw_scenes")

    class Config:
        env_file = ".env"

config = Settings()
