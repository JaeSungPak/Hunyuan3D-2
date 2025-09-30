# # client/client.py

# import requests

# SERVER_URL = "http://localhost:8888"  # 또는 포트포워딩 주소
# PROMPT = "barbie-style chairs and tables"

# def request_model(prompt, save_as):
#     response = requests.get(f"{SERVER_URL}/generate", params={"prompt": prompt})
#     if response.status_code == 200 and response.headers["content-type"] == "model/gltf-binary":
#         with open(save_as, "wb") as f:
#             f.write(response.content)
#         print(f"✅ GLB 파일 저장 완료: {save_as}")
#     else:
#         print("❌ 오류 발생:", response.json())

# if __name__ == "__main__":
#     filename = PROMPT.replace(" ", "_") + ".glb"
#     request_model(PROMPT, filename)

# client/client.py

import argparse
import requests
import os

SERVER_URL = "http://localhost:8000"  # 또는 포트포워딩 주소

def request_model(prompt, save_as):
    response = requests.get(f"{SERVER_URL}/generate", params={"prompt": prompt})
    if response.status_code == 200 and response.headers["content-type"] == "model/gltf-binary":
        os.makedirs(os.path.dirname(save_as), exist_ok=True)
        with open(save_as, "wb") as f:
            f.write(response.content)
        print(f"✅ GLB 파일 저장 완료: {save_as}")
    else:
        try:
            print("❌ 오류 발생:", response.json())
        except Exception:
            print("❌ 응답 파싱 실패")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="서버에 텍스트 프롬프트를 전송하고 GLB 파일을 저장합니다.")
    parser.add_argument("prompt", type=str, help="3D 모델을 생성할 텍스트 프롬프트")
    parser.add_argument("save_path", type=str, help="저장할 GLB 파일 경로 (예: ./output/barbie.glb)")
    args = parser.parse_args()

    request_model(args.prompt, args.save_path)
