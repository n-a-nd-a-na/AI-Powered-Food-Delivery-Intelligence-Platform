From python:3.11

WORKDIR /app

COPY Frontend/requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY Frontend/ Frontend/

CMD ["streamlit","run","Frontend/app.py","--server.address=0.0.0.0","--server.port=8501"]