# server/main.py

import os
import torch
from PIL import Image
from fastapi import FastAPI, Query
from fastapi.responses import FileResponse, JSONResponse
from rembg import remove
from diffusers import StableDiffusionPipeline
from hy3dgen.texgen import Hunyuan3DPaintPipeline
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
from hy3dgen.shapegen import FaceReducer, FloaterRemover, DegenerateFaceRemover

app = FastAPI()

# 사전 로딩 (모델 로딩은 매우 느리므로 서버 시작 시 한번만)
print("🔧 모델 로딩 중...")
text2img_pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16
).to("cuda")

shape_pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained("tencent/Hunyuan3D-2")
paint_pipe = Hunyuan3DPaintPipeline.from_pretrained("tencent/Hunyuan3D-2")

cleaners = [FloaterRemover(), DegenerateFaceRemover(), FaceReducer()]
print("✅ 모델 로딩 완료!")

@app.get("/generate")
def generate_model(prompt: str = Query(..., description="텍스트 프롬프트로 3D 모델 생성")):
    # 경로 및 파일 설정
    safe_prompt = prompt.replace(" ", "_")
    glb_path = f"logs/{safe_prompt}.glb"
    img_path = "assets/demo.png"

    # 캐시된 모델이 있다면 바로 반환
    if os.path.exists(glb_path):
        return FileResponse(glb_path, media_type="model/gltf-binary", filename=os.path.basename(glb_path))

    try:
        # 1. 텍스트 → 이미지 생성
        full_prompt = f"{prompt}, centered in the frame"
        image = text2img_pipe(full_prompt).images[0]

        # 2. 배경 제거
        image_no_bg = remove(image)

        os.makedirs("assets", exist_ok=True)
        image_no_bg.save(img_path)
        print("🖼️ 이미지 생성 및 배경 제거 완료")

        # 3. 3D 형태 생성
        mesh = shape_pipe(image=img_path)[0]
        for cleaner in cleaners:
            mesh = cleaner(mesh)
        print("📐 메쉬 생성 완료")

        # 4. 텍스처 입히기
        mesh = paint_pipe(mesh, image=img_path)
        print("🎨 텍스처 생성 완료")

        # 5. GLB 파일 저장
        os.makedirs("logs", exist_ok=True)
        mesh.export(glb_path)
        print("💾 GLB 파일 저장 완료")

        return FileResponse(glb_path, media_type="model/gltf-binary", filename=os.path.basename(glb_path))

    except Exception as e:
        print("❌ 오류 발생:", e)
        return JSONResponse(status_code=500, content={"error": "3D 모델 생성 중 오류 발생", "details": str(e)})
