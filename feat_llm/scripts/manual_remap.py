"""
수동 메타데이터 매핑 업데이트 스크립트
- 인덱스 스키마에 body_region(Edm.String), pain_area(Collection) 필드 추가
- 11개 무릎 청크에 값 적용
실행: python feat_llm/scripts/manual_remap.py
"""
import os
from pathlib import Path
from dotenv import load_dotenv
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchField, SearchFieldDataType, SimpleField,
)

load_dotenv(dotenv_path=Path(__file__).parent.parent.parent / ".env")

endpoint  = os.getenv("AZURE_AI_SEARCH_ENDPOINT")
key       = os.getenv("AZURE_AI_SEARCH_KEY")
INDEX     = "taping-guide-index-3"

index_client = SearchIndexClient(endpoint=endpoint, credential=AzureKeyCredential(key))
doc_client   = SearchClient(endpoint=endpoint, index_name=INDEX, credential=AzureKeyCredential(key))

# ── 1. 스키마에 body_region / pain_area 필드 추가 ──────────────
print("[1] 스키마 업데이트 중...")
index = index_client.get_index(INDEX)

existing = {f.name for f in index.fields}
new_fields = []

if "body_region" not in existing:
    new_fields.append(SimpleField(
        name="body_region",
        type=SearchFieldDataType.String,
        filterable=True,
    ))
    print("  + body_region (Edm.String, filterable) 추가")
else:
    print("  - body_region 이미 존재, 스킵")

if "pain_area" not in existing:
    new_fields.append(SearchField(
        name="pain_area",
        type=SearchFieldDataType.Collection(SearchFieldDataType.String),
        filterable=True,
    ))
    print("  + pain_area (Collection(Edm.String), filterable) 추가")
else:
    print("  - pain_area 이미 존재, 스킵")

if new_fields:
    index.fields.extend(new_fields)
    index_client.create_or_update_index(index)
    print("  스키마 업데이트 완료")
else:
    print("  변경 없음")

# ── 2. 매핑 테이블 ─────────────────────────────────────────────
MAPPING = [
    # 1
    {"chunk_id": "bcf629ac-94fd-4359-b9ee-271b2abb0889",
     "technique_code": "KT_KNEE_FULL", "condition": "PFPS",
     "body_region": "anterior", "pain_area": ["inferior", "anterior"]},
    # 2
    {"chunk_id": "fc32754d-8488-405e-a20d-b9669c4a6805",
     "technique_code": "KT_KNEE_LATERAL", "condition": "Lateral",
     "body_region": "lateral", "pain_area": ["lateral", "anterior"]},
    # 3
    {"chunk_id": "09dbf695-69ff-447d-85dd-ed5c030861c4",
     "technique_code": "KT_KNEE_MALALIGNMENT", "condition": "Malalignment",
     "body_region": "anterior", "pain_area": ["anterior"]},
    # 4
    {"chunk_id": "89c0ff7f-5e1f-4ed6-8e6e-1f8e8936122e",
     "technique_code": "KT_KNEE_FULL", "condition": "Full Knee",
     "body_region": "anterior", "pain_area": ["inferior", "anterior"]},
    # 5
    {"chunk_id": "2e36a9d0-ce3d-4b14-9554-2f9a08f11669",
     "technique_code": "KT_KNEE_MALALIGNMENT", "condition": "Malalignment",
     "body_region": "anterior", "pain_area": ["anterior"]},
    # 6 — region 오태깅 수정 포함
    {"chunk_id": "1e0cb71e-c379-442d-bdad-92eb3d87b028",
     "technique_code": "KT_KNEE_MEDIAL", "condition": "Medial",
     "body_region": "medial", "pain_area": ["medial"], "region": "medial"},
    # 7
    {"chunk_id": "40ab8d89-cf41-4082-ab05-2951142af914",
     "technique_code": "KT_KNEE_MEDIAL", "condition": "Medial",
     "body_region": "medial", "pain_area": ["medial"]},
    # 8
    {"chunk_id": "2fcaeeb7-5949-4ce5-acf5-5b628907ec58",
     "technique_code": "KT_KNEE_GENERAL", "condition": "General",
     "body_region": "anterior", "pain_area": ["anterior", "kneecap"]},
    # 9
    {"chunk_id": "b062fc2e-f9b1-414e-b928-cffd8a708565",
     "technique_code": "KT_KNEE_GENERAL", "condition": "General",
     "body_region": "anterior", "pain_area": ["anterior", "kneecap"]},
    # 10
    {"chunk_id": "50ca1d30-dbc1-4b80-814b-11b4fbb34ccd",
     "technique_code": "KT_KNEE_MALALIGNMENT", "condition": "Malalignment",
     "body_region": "anterior", "pain_area": ["anterior"]},
    # 11
    {"chunk_id": "f0337a32-5ef1-41b1-ac41-4aa7b24f26b4",
     "technique_code": "KT_KNEE_LATERAL", "condition": "Lateral",
     "body_region": "lateral", "pain_area": ["lateral", "anterior"]},
]

# ── 3. 문서 업데이트 ───────────────────────────────────────────
print(f"\n[2] 문서 업데이트 중... ({len(MAPPING)}개)")
results = doc_client.merge_or_upload_documents(documents=MAPPING)

success = sum(1 for r in results if r.succeeded)
fail    = sum(1 for r in results if not r.succeeded)
print(f"  성공: {success}개 / 실패: {fail}개")
for r in results:
    status = "OK" if r.succeeded else f"FAIL({r.error_message})"
    print(f"  {r.key[:8]}...  {status}")
