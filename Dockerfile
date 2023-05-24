FROM python:3.9

WORKDIR /app

COPY requirements.txt .

RUN pip install accelerate>=0.12.0 transformers[torch]==4.25.1 fastapi uvicorn gunicorn

COPY . .

EXPOSE 8000

CMD ["gunicorn", "--bind", "0.0.0.0:8000", "app:app"]
