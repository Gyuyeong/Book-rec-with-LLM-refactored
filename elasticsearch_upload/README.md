# Elasticsearch 데이터 업로더

이 스크립트는 CSV 파일에서 Elasticsearch 인덱스로 데이터를 업로드하기 위해 설계되었습니다. `sentence-transformers` 및 `elasticsearch` 파이썬 패키지를 사용하여 데이터를 효율적으로 처리하고 업로드합니다.

## 필요 조건

- Python 3.6 이상
- Elasticsearch 인스턴스
- `sentence-transformers`
- `elasticsearch`
- `pandas`
- `tqdm`

## 설치

스크립트를 실행하기 전에 필요한 패키지가 설치되어 있는지 확인하세요. pip를 사용하여 설치할 수 있습니다:

```bash
pip install sentence-transformers elasticsearch pandas tqdm
```

## 설정

1. **Elasticsearch 연결**: 스크립트에서 Elasticsearch 인스턴스 세부 정보를 업데이트하세요.

   ```python
   es = Elasticsearch(
       ["https://your_elasticsearch_server:port"],
       basic_auth=("username", "password"),
       verify_certs=False,
       timeout=30,
       max_retries=10,
       retry_on_timeout=True,
   )
   ```

2. **입력 파일**: CSV 파일의 경로를 지정하세요.

   ```python
   input_filename = "path_to_your_csv_file"
   ```

3. **인덱스 설정**: 스크립트는 미리 정의된 인덱스 설정을 사용합니다. 필요에 따라 `setting` 및 `mapping` 변수를 수정하세요.

## 사용법

Python 환경에서 스크립트를 실행하세요. 스크립트는 CSV 파일을 읽고, 데이터를 처리한 후 지정된 Elasticsearch 인덱스에 청크 단위로 업로드합니다.

```bash
python your_script_name.py
```

## 기능

- CSV 파일에서 데이터를 읽습니다.
- Elasticsearch 문서를 생성하기 위해 각 행을 처리합니다.
- `sentence-transformers`를 사용하여 텍스트에 대한 임베딩을 생성합니다.
- 청크 단위로 Elasticsearch에 문서를 업로드합니다.

## 주의 사항

- CSV 파일이 올바른 형식과 인코딩(UTF-8)인지 확인하세요.
- Elasticsearch 인스턴스의 건강과 가용성을 확인하세요.
- 오류나 경고에 대해 스크립트의 진행 상황과 로그를 모니터링하세요.

---
