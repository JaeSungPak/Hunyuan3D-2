# server_object.py

import os
import uuid
import torch
from PIL import Image
from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from rembg import remove
from diffusers import StableDiffusionPipeline
from hy3dgen.texgen import Hunyuan3DPaintPipeline
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
from hy3dgen.shapegen import FaceReducer, FloaterRemover, DegenerateFaceRemover

# --- 초기 설정 ---
app = FastAPI()

# 작업 상태를 저장할 글로벌 변수
# In-memory storage; for production, use Redis or a database.
task_status = {}

print("🔧 모델 로딩 중...")
# 모델 로딩은 서버 시작 시 한번만 수행됩니다.
text2img_pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16
).to("cuda")
shape_pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained("tencent/Hunyuan3D-2")
paint_pipe = Hunyuan3DPaintPipeline.from_pretrained("tencent/Hunyuan3D-2")
cleaners = [FloaterRemover(), DegenerateFaceRemover(), FaceReducer()]
print("✅ 모델 로딩 완료!")


# --- 3D 모델 생성 핵심 함수 (백그라운드에서 실행) ---
def generate_3d_model_background(task_id: str, prompt: str):
    """
    실제 3D 모델을 생성하는 함수. 각 단계마다 task_status를 업데이트합니다.
    """
    try:
        safe_prompt = prompt.replace(" ", "_")
        glb_path = f"logs/{safe_prompt}_{task_id[:8]}.glb"
        img_path = f"assets/img_{task_id[:8]}.png"
        
        # 0. 경로 생성
        os.makedirs("logs", exist_ok=True)
        os.makedirs("assets", exist_ok=True)

        # 1. 텍스트 → 이미지 생성
        task_status[task_id] = {"status": "🖼️ 1/5: 텍스트에서 이미지 생성 중...", "progress": 20, "file_path": None}
        full_prompt = f"{prompt}, centered in the frame"
        image = text2img_pipe(full_prompt, num_inference_steps=30).images[0]

        # 2. 배경 제거
        task_status[task_id] = {"status": "✨ 2/5: 이미지 배경 제거 중...", "progress": 40, "file_path": None}
        image_no_bg = remove(image)
        image_no_bg.save(img_path)

        # 3. 3D 형태 생성
        task_status[task_id] = {"status": "📐 3/5: 3D 메쉬 생성 중...", "progress": 60, "file_path": None}
        mesh = shape_pipe(image=img_path)[0]
        for cleaner in cleaners:
            mesh = cleaner(mesh)

        # 4. 텍스처 입히기
        task_status[task_id] = {"status": "🎨 4/5: 텍스처 적용 중...", "progress": 80, "file_path": None}
        mesh = paint_pipe(mesh, image=img_path)

        # 5. GLB 파일 저장
        task_status[task_id] = {"status": "💾 5/5: GLB 파일 저장 중...", "progress": 95, "file_path": None}
        mesh.export(glb_path)

        # 완료
        task_status[task_id] = {"status": "✅ 생성 완료!", "progress": 100, "file_path": glb_path}
        print(f"✅ 작업 완료 (ID: {task_id}): {glb_path}")

    except Exception as e:
        print(f"❌ 오류 발생 (ID: {task_id}): {e}")
        task_status[task_id] = {"status": f"❌ 오류 발생: {e}", "progress": 0, "file_path": None}


# --- API 엔드포인트 ---
@app.post("/generate")
def generate_model_async(prompt: str, background_tasks: BackgroundTasks):
    """
    모델 생성 요청을 받고, 백그라운드 작업을 시작시킨 후 작업 ID를 반환합니다.
    """
    task_id = str(uuid.uuid4())
    task_status[task_id] = {"status": "⏳ 0/5: 작업 대기 중...", "progress": 0, "file_path": None}
    
    # 캐시 확인 (간단한 버전)
    safe_prompt = prompt.replace(" ", "_")
    # 실제 프로덕션에서는 더 정교한 캐시 키 관리 필요
    cached_files = [f for f in os.listdir("logs") if f.startswith(safe_prompt) and f.endswith(".glb")]
    if cached_files:
        cached_file_path = os.path.join("logs", cached_files[0])
        task_status[task_id] = {"status": "✅ 캐시된 파일 발견!", "progress": 100, "file_path": cached_file_path}
        return {"task_id": task_id}
        
    background_tasks.add_task(generate_3d_model_background, task_id, prompt)
    return {"task_id": task_id}


@app.get("/status/{task_id}")
def get_status(task_id: str):
    """
    주어진 작업 ID의 현재 상태를 반환합니다.
    """
    status = task_status.get(task_id)
    if not status:
        return JSONResponse(status_code=404, content={"error": "Task not found"})
    return status

@app.get("/download/{task_id}")
def download_file(task_id: str):
    """
    완료된 작업의 GLB 파일을 다운로드합니다.
    """
    status = task_status.get(task_id)
    if not status or status['progress'] != 100 or not status.get('file_path'):
        return JSONResponse(status_code=404, content={"error": "File not ready or not found"})
    
    file_path = status['file_path']
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type="model/gltf-binary", filename=os.path.basename(file_path))
    else:
        return JSONResponse(status_code=404, content={"error": "File not found on disk"})