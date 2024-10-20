# Better use newer Python as generated code can use new features
FROM python:3.11-slim

# install git and c++
RUN apt-get update && apt-get install -y git g++

# upgrade to latest pip
RUN pip install --upgrade pip

COPY . /evalplus

RUN cd /evalplus && pip install ".[perf]"

# Test cirronlib.cpp which invokes compilation on first use
RUN python3 -c 'from cirron import Collector; exec("with Collector(): pass")'

# Pre-install the dataset
RUN python3 -c "from evalplus.data import *; get_human_eval_plus(); get_mbpp_plus(); get_evalperf_data()"

WORKDIR /app

CMD ["bash"]
