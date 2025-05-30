import sys
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch
import librosa
import numpy as np
import io

app = FastAPI(title="Audio Transcription API")

# إضافة دعم CORS للسماح بالطلبات من أي مصدر
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# تحميل المعالج والنموذج
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h")

async def transcribe_audio(audio_data: bytes):
    try:
        # تحميل الصوت باستخدام librosa من البايتات
        audio_input, _ = librosa.load(io.BytesIO(audio_data), sr=16000)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"خطأ في تحميل الصوت: {str(e)}")

    # معالجة الصوت
    inputs = processor(audio_input, return_tensors="pt", padding=True)

    # تنفيذ النسخ
    with torch.no_grad():
        logits = model(input_values=inputs.input_values).logits

    # فك تشفير المعرفات المتوقعة إلى نص
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.decode(predicted_ids[0])
    return transcription

@app.post("/transcribe", response_model=dict)
async def transcribe(file: UploadFile = File(...)):
    # التحقق من أن الملف المرفوع هو ملف صوتي
    if not file.content_type.startswith("audio/"):
        raise HTTPException(status_code=400, detail="يجب أن يكون الملف المرفوع ملفًا صوتيًا")

    try:
        # قراءة محتوى الملف الصوتي
        audio_data = await file.read()
        transcription = await transcribe_audio(audio_data)
        return {"transcription": transcription}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"خطأ في النسخ: {str(e)}")

@app.get("/")
async def root():
    return {"message": "مرحبًا بكم في API استخراج النصوص الصوتية. استخدم POST /transcribe لرفع ملف صوتي."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)