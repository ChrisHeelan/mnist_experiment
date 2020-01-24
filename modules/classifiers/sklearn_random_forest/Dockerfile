FROM continuumio/miniconda3:4.7.12

WORKDIR /home/sklearn_random_forest

ADD modules/classifiers/sklearn_random_forest/requirements.txt /home/sklearn_random_forest/requirements.txt
RUN conda install --file /home/sklearn_random_forest/requirements.txt

COPY modules/classifiers/sklearn_random_forest /home/sklearn_random_forest

RUN apt-get update && apt-get install -y git
RUN git clone https://github.com/ChrisHeelan/base_dockex.git
RUN git -C /home/sklearn_random_forest/base_dockex checkout refactor_BaseModel

ENV PYTHONPATH /home/sklearn_random_forest

ENTRYPOINT ["python", "-u", "/home/sklearn_random_forest/sklearn_random_forest.py"]
