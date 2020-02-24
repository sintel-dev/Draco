FROM python:3.6

ARG UID=1000
EXPOSE 8888

RUN adduser jupyter --uid $UID --disabled-password --system

RUN mkdir /app
COPY setup.py /app
RUN mkdir /app/greenguard
COPY greenguard/__init__.py /app/greenguard
RUN pip install -e /app jupyter

RUN rm -r /app/greenguard
COPY greenguard /app/greenguard
COPY notebooks /app/notebooks

WORKDIR /app
USER jupyter
CMD /usr/local/bin/jupyter notebook --ip 0.0.0.0 --NotebookApp.token=''
