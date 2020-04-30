FROM tensorflow/tensorflow:1.14.0-gpu-py3

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update
RUN apt-get install -y --no-install-recommends \
       git \
       wget \
       python-tk
RUN git clone https://github.com/AnonymousBMVA/input_channel_dropout.git
WORKDIR input_channel_dropout
#RUN wget https://1drv.ms/u/s!AkO079ItTrSpaByU2MyIs0WeNPM?e=guajrc
RUN pip install Cython contextlib2 jupyter matplotlib pillow lxml
RUN git clone https://github.com/cocodataset/cocoapi.git
WORKDIR cocoapi/PythonAPI
RUN make
RUN cp -r pycocotools ../../
WORKDIR ../../
ENV PYTHONPATH $PYTHONPATH:`pwd`:`pwd`/research
ENV PYTHONPATH $PYTHONPATH:`pwd`:`pwd`/research/slim


