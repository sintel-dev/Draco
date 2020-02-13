FROM python:3.6

EXPOSE 8888

RUN adduser jupyter --uid 1000 --disabled-password --system

RUN mkdir /greenguard
COPY setup.py /greenguard
RUN pip install -e /greenguard && pip install jupyter

COPY greenguard /greenguard/greenguard
COPY notebooks /greenguard/notebooks

WORKDIR /greenguard
USER jupyter
CMD /usr/local/bin/jupyter notebook --ip 0.0.0.0 --NotebookApp.token=''
