FROM python:3.7-slim

WORKDIR /python

RUN pip3 install grpcio grpcio-tools protobuf torch torchvision matplotlib

COPY . .

EXPOSE 50051
CMD [ "python3", "server.py"]