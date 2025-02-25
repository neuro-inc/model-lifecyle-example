FROM python:3.13

WORKDIR /app

COPY requirements.txt .
RUN pip install --progress-bar=off -U --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu126

# COPY modules/ modules/
# COPY train.py train.py
# COPY char_rnn_classification_tutorial.py char_rnn_classification_tutorial.py

# RUN python train.py
