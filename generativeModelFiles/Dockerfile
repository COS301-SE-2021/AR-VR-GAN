FROM python:3.8-slim-buster

WORKDIR /python

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt --use-feature=2020-resolver

COPY . .

CMD [ "python3", "server.py"]