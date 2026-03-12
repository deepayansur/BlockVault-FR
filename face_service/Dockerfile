FROM python:3.11-slim

ARG ORT_PACKAGE=onnxruntime
RUN apt-get update && apt-get install -y --no-install-recommends build-essential libgl1 libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY face_service/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt && pip install --no-cache-dir ${ORT_PACKAGE}
COPY face_service/app.py ./app.py
COPY face_service ./face_service
COPY face_service/face_db ./face_db
EXPOSE 8001
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8001"]
