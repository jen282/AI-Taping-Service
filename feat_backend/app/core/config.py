from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    COSMOS_ENDPOINT: str
    COSMOS_KEY: str
    COSMOS_DATABASE: str = "TapingDB"
    SESSION_CONTAINER: str = "Sessions"
    REGISTRY_CONTAINER: str = "TapingRegistry"
    
    # 이 줄을 추가해 주세요!
    AZURE_STORAGE_CONNECTION_STRING: str 

    class Config:
        env_file = ".env"

settings = Settings()