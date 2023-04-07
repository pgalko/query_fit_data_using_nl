FROM python:3.10

WORKDIR /tmp
RUN python -m pip install --upgrade pip

COPY requirements.txt /tmp/
RUN python -m pip install -r requirements.txt

RUN mkdir /app
WORKDIR /app
COPY ./ /app/

EXPOSE 80
ENV PORT 80

COPY ./ /app

CMD uvicorn main:app --host 0.0.0.0 --port 80 --reload
