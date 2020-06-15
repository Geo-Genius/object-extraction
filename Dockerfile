FROM swr.cn-north-5.myhuaweicloud.com/geogenius_dev/geogenius_sdk:v1.0
ADD ./object-extraction.tar.gz /opt
ENV PATH ${PATH}:/root/anaconda3/bin/
WORKDIR /opt/object-extraction
RUN ["/root/anaconda3/bin/python", "setup.py", "install"]
