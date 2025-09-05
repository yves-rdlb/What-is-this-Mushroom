FROM python:3.12.9

COPY MUSHROOM /MUSHROOM
COPY requirements.txt /requirements.txt
COPY model_main /model_main

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

CMD uvicorn MUSHROOM.api.fast:app --host 0.0.0.0 --port $PORT
