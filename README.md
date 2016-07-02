This project requires the following libraries for visualizations:
* graphviz
* pydot

A docker container has been prepared with the needed libraries:
```bash
docker run -dt -p 8888:8888 -v $WORKDIR:/home/jovyan/work sjawhar/udacity_p5
```