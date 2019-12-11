FROM continuumio/miniconda3:4.7.12

WORKDIR /home/sklearn_classification_report

ADD modules/analysis/sklearn_classification_report/requirements.txt /home/sklearn_classification_report/requirements.txt
RUN conda install --file /home/sklearn_classification_report/requirements.txt

COPY modules/analysis/sklearn_classification_report /home/sklearn_classification_report

RUN git clone https://github.com/ChrisHeelan/base_dockex.git

ENV PYTHONPATH /home/sklearn_classification_report

ENTRYPOINT ["python", "-u", "/home/sklearn_classification_report/sklearn_classification_report.py"]
