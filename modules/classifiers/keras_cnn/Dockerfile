FROM tensorflow/tensorflow:latest-gpu-py3

WORKDIR /home/keras_cnn

ADD modules/classifiers/keras_cnn/requirements.txt /home/keras_cnn
RUN pip install -r /home/keras_cnn/requirements.txt

COPY modules/classifiers/keras_cnn/ /home/keras_cnn

RUN apt-get update && apt-get install -y git
RUN git clone https://github.com/ChrisHeelan/base_dockex.git
RUN git -C /home/keras_cnn/base_dockex checkout refactor_BaseModel

ENV PYTHONPATH /home/keras_cnn

ENTRYPOINT ["python", "-u", "/home/keras_cnn/keras_cnn.py"]
