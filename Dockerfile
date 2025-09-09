FROM python:3.12.9

COPY MUSHROOM /MUSHROOM
COPY requirements.txt /requirements.txt

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

CMD uvicorn MUSHROOM.api.VIT_API:app --host 0.0.0.0 --port $PORT
