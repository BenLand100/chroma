FROM nruof/chroma3:nvidia.base
LABEL Nicholas Ruof <nickruof@uw.edu>

WORKDIR /opt/
ADD https://api.github.com/repos/nickruof/chroma/git/refs/heads/master /opt/chroma_version.json
RUN git clone https://github.com/nickruof/chroma
WORKDIR /opt/chroma/
RUN sed -i 's/VIRTUAL_ENV/CONDA_PREFIX/g' setup.py #use anaconda env instead
RUN python setup.py develop

#For jupyter
EXPOSE 8888

CMD ["/bin/bash","-l"]
