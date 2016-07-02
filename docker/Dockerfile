FROM jupyter/datascience-notebook
MAINTAINER Sami Jawhar <sami.n.jawhar@gmail.com>

USER root
RUN apt-get update --fix-missing && \
    apt-get install -y graphviz

USER $NB_USER
RUN conda install -y -n python2 pydot
