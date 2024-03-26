FROM python:3.8

RUN mkdir -p /app/upload
RUN mkdir -p /app/result

COPY . .
COPY . /app

RUN pip install opencv-python==4.9.0.80
RUN pip install torchvision==0.13.0
RUN pip install torch==1.12.0
RUN apt-get update && apt-get install -y libgl1-mesa-glx
RUN pip install tqdm
RUN pip install flask==3.0.2
RUN pip install flask_cors
RUN pip install pandas==2.0.3
RUN pip install pyyaml==6.0.1
RUN pip install ultralytics==8.1.5

WORKDIR /app

EXPOSE 5002

CMD ["python", "main.py"]

