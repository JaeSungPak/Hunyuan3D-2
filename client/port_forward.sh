#!/usr/bin/env bash

# 1. load.py 실행해서 포트번호 가져오기
PORT_NUM=$(python3 load_docs.py)

# 2. 로드된 포트번호 표시
echo "Loaded port number: $PORT_NUM"

# 3. SSH 포트포워딩 실행
ssh -L 8888:localhost:8000 -p "$PORT_NUM" root@143.248.249.104
