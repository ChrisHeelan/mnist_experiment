FROM continuumio/miniconda3:4.7.12

WORKDIR /home/load_mnist

ADD modules/data/load_mnist/requirements.txt /home/load_mnist/requirements.txt
RUN conda install --file /home/load_mnist/requirements.txt

COPY modules/data/load_mnist /home/load_mnist

RUN git clone https://github.com/ChrisHeelan/base_dockex.git

ENV PYTHONPATH /home/load_mnist

ENTRYPOINT ["python", "-u", "/home/load_mnist/load_mnist.py"]
