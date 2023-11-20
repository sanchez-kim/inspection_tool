# Inspection Tool
- 데이터 검수 도구 (내부용)
- remote ip, password 등 민감한 정보는 `.env`에 따로 설정해야 함.

## Codes
- `main.py`: 메인 코드
- `render_test.py`: 단일 obj 파일에 대한 렌더링 테스트용
- `missing_files.py`: 원본데이터의 목록을 사용하여, 다운로드된 파일 중에 누락된 것이 있는지 체크
- `run.py`: 코드가 성공적으로 실행될 때까지 스크립트를 계속 실행
- `ssh.py`: remote server로부터 데이터 다운로드