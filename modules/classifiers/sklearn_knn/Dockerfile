FROM continuumio/miniconda3:4.7.12

WORKDIR /home/sklearn_knn

ADD modules/classifiers/sklearn_knn/requirements.txt /home/sklearn_knn/requirements.txt
RUN conda install --file /home/sklearn_knn/requirements.txt

COPY modules/classifiers/sklearn_knn /home/sklearn_knn

RUN apt-get update && apt-get install -y git
RUN git clone https://github.com/ChrisHeelan/base_dockex.git
RUN git -C /home/sklearn_knn/base_dockex checkout refactor_BaseModel

ENV PYTHONPATH /home/sklearn_knn

ENTRYPOINT ["python", "-u", "/home/sklearn_knn/sklearn_knn.py"]
