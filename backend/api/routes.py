import io
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from PIL import Image

from backend.models.presets import VALID_PRESETS
from backend.services.beautify import beautify_image

router = APIRouter(prefix="/api")


@router.post("/beauty")
async def beauty_endpoint(
    image: UploadFile = File(...),
    preset: str = Form(...),
):
    preset = preset.lower()
    if preset not in VALID_PRESETS:
        raise HTTPException(status_code=400, detail="Preset tidak dikenal")

    try:
        contents = await image.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="File gambar tidak valid")

    out_img = beautify_image(img, preset)

    buffer = io.BytesIO()
    out_img.save(buffer, format="JPEG", quality=95)
    buffer.seek(0)

    return Response(
        content=buffer.read(),
        media_type="image/jpeg"
    )
