docker build -t owabench ./Docker/
docker run -it -p 8888:8888 -v $(pwd):/home/jovyan/work owabench

