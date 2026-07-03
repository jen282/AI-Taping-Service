"""
fetch_raw_data.py — Azure Blob Storage에서 교본 txt 파일을 data/raw/에 다운로드

Storage account : tapingdata1
Container       : taping-guide-processed
Blob prefix     : kine-book-data-test/rag/

Usage:
    python fetch_raw_data.py
"""

import os
from pathlib import Path

from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv

CONTAINER_NAME = "taping-guide-processed"
BLOB_PREFIX    = "kine-book-data-test/rag/"

def main() -> None:
    env_path = Path(__file__).parent.parent.parent / ".env"
    load_dotenv(env_path)

    conn_str = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    if not conn_str:
        raise EnvironmentError("AZURE_STORAGE_CONNECTION_STRING 환경 변수가 없습니다.")

    raw_dir = Path(__file__).parent.parent / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    client = BlobServiceClient.from_connection_string(conn_str)
    container = client.get_container_client(CONTAINER_NAME)

    blobs = [b for b in container.list_blobs(name_starts_with=BLOB_PREFIX)
             if b.name.endswith(".txt")]

    if not blobs:
        print(f"[fetch] {CONTAINER_NAME}/{BLOB_PREFIX} 에서 .txt 파일을 찾을 수 없습니다.")
        return

    print(f"[fetch] {len(blobs)}개 파일 발견 → {raw_dir}")

    for blob in blobs:
        filename = Path(blob.name).name
        dest = raw_dir / filename

        if dest.exists():
            print(f"  skip (already exists): {filename}")
            continue

        data = container.download_blob(blob.name).readall()
        dest.write_bytes(data)
        print(f"  downloaded: {filename}  ({len(data):,} bytes)")

    print("[fetch] Done.")


if __name__ == "__main__":
    main()
