FROM python:3.10
WORKDIR //Users/jedaelisbon/Documents/Projects/Email-Classifier/
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
# COPY main.py ./work_dir.py
RUN python -m nltk.downloader punkt
RUN python -m nltk.downloader stopwords
EXPOSE 8501
CMD ["streamlit","run","app.py","--server.port=8501","--server.address=0.0.0.0"]
