# syntax=docker/dockerfile:1

FROM quay.io/jupyter/base-notebook

USER root

COPY requirements.txt ./requirements.txt

RUN apt-get clean
RUN apt-get update
RUN apt-get install -y libglib2.0-0 libglib2.0-dev
RUN apt-get install -y libgl1

RUN pip install --no-cache-dir -r requirements.txt