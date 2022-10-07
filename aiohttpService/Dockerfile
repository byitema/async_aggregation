FROM python:3.8-slim-buster

WORKDIR /code

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

COPY server.py server.py
COPY config.ini config.ini

CMD ["python3", "server.py"]