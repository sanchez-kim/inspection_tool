# Inspection Tool

- 데이터 검수 도구 (내부용)
- remote ip, password 등의 정보는 `.env`에 따로 설정해야 함.

## Codes

- `main.py`: 메인 코드
- `render_test.py`: 단일 obj 파일에 대한 렌더링 테스트용
- `utils/missing_files.py`: 원본데이터의 목록을 사용하여, 다운로드된 파일 중에 누락된 것이 있는지 체크
- `run.py`: 코드가 성공적으로 실행될 때까지 스크립트를 계속 실행
- `ssh.py`: remote server로부터 데이터 다운로드

## Inspection Steps

```bash
# 서버로부터 파일 요청 및 다운로드
python main.py --m [model_num] --download
# 랜드마크 계산
python main.py -m [model_num] --eval
# 랜드마크 평균 계산
python average.py -m [model_num]
# 랜드마크 기준 pass / fail 판별
cd statistics
python pass_fail.py -a [path/to/log_analysis_output.txt]
```
