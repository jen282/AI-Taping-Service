from azure.storage.blob.aio import BlobServiceClient
from app.core.config import settings

async def upload_file_to_blob(file, blob_name: str):
    blob_service_client = BlobServiceClient.from_connection_string(settings.AZURE_STORAGE_CONNECTION_STRING)
    container_client = blob_service_client.get_container_client("user-uploads")
    
    # 컨테이너 없으면 생성 (최초 1회)
    await container_client.create_container_if_not_exists()
    
    blob_client = container_client.get_blob_client(blob_name)
    content = await file.read()
    await blob_client.upload_blob(content, overwrite=True)
    
    return blob_client.url