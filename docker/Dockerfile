FROM python:3.6

ARG UID=1000
EXPOSE 8888

RUN mkdir /app
COPY setup.py /app
COPY draco /app/draco
COPY tutorials /app/tutorials
RUN pip install -e /app jupyter

WORKDIR /app
CMD pip install -e /app && /usr/local/bin/jupyter notebook --ip 0.0.0.0 --NotebookApp.token='' --allow-root
