FROM python:3.11-slim
WORKDIR /app
COPY backend /app
RUN pip install --no-cache-dir uvicorn && pip install --no-cache-dir -e .
ENV PYTHONPATH=/app
CMD ["uvicorn","app.main:app","--host","0.0.0.0","--port","8080"]
