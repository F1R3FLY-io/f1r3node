FROM python:3.11-slim

RUN pip install --no-cache-dir \
    transformers \
    torch --extra-index-url https://download.pytorch.org/whl/cpu

WORKDIR /sbert-downloader
COPY downloader.py .

CMD ["python", "downloader.py"]