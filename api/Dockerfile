FROM python:3.11-slim

WORKDIR /home/rayane/machine-learning-projects/api

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py vectorizer.pkl spam_model.pkl ./

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
