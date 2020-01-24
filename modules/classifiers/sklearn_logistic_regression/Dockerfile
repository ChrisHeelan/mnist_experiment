FROM continuumio/miniconda3:4.7.12

WORKDIR /home/sklearn_logistic_regression

ADD modules/classifiers/sklearn_logistic_regression/requirements.txt /home/sklearn_logistic_regression/requirements.txt
RUN conda install --file /home/sklearn_logistic_regression/requirements.txt

COPY modules/classifiers/sklearn_logistic_regression /home/sklearn_logistic_regression

RUN apt-get update && apt-get install -y git
RUN git clone https://github.com/ChrisHeelan/base_dockex.git
RUN git -C /home/sklearn_logistic_regression/base_dockex checkout refactor_BaseModel

ENV PYTHONPATH /home/sklearn_logistic_regression

ENTRYPOINT ["python", "-u", "/home/sklearn_logistic_regression/sklearn_logistic_regression.py"]
