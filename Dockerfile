FROM python:3.9

COPY requirements.txt .

RUN pip3 install -r requirements.txt

RUN pip install gunicorn

RUN mkdir /app
WORKDIR /app

COPY proto/inference.proto /app/proto/
COPY server.py /app/
COPY tests.py /app/
COPY grpc_server.py /app/
COPY setup.py /app/

RUN python3 -m grpc_tools.protoc --pyi_out=. -I./proto --grpc_python_out=. --python_out=. ./proto/inference.proto

COPY supervisord.conf /etc/supervisord.conf

ENTRYPOINT ["bash", "-c", "python3 setup.py && supervisord -c /etc/supervisord.conf"]