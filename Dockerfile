FROM python:3.8

COPY . .
COPY . /app


RUN pip install opencv-python==4.9.0.80
RUN pip install torchvision==0.13.0
RUN apt-get update && apt-get install -y libgl1-mesa-glx
RUN pip install tqdm
RUN pip install flask==3.0.2

WORKDIR /app

CMD ["python", "main.py"]

