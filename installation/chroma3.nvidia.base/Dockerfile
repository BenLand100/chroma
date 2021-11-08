FROM nvidia/cudagl:10.1-devel-ubuntu18.04 as build-anaconda
LABEL Nicholas Ruof <nickruof@uw.edu>

COPY chroma_env.sh /etc/profile.d/
SHELL ["/bin/bash", "-l", "-c"] 
RUN apt-get update && apt-get install -y curl git

WORKDIR /opt/
RUN curl -O https://repo.anaconda.com/archive/Anaconda3-2021.05-Linux-x86_64.sh
RUN chmod +x Anaconda3-2021.05-Linux-x86_64.sh
RUN ./Anaconda3-2021.05-Linux-x86_64.sh -b -p /opt/anaconda3
RUN rm Anaconda3-2021.05-Linux-x86_64.sh
RUN conda install -y -c anaconda boost cmake

FROM nvidia/cudagl:10.1-devel-ubuntu18.04  AS build-root
LABEL Nicholas Ruof <nickruof@uw.edu>

USER root
COPY chroma_env.sh /etc/profile.d/
SHELL ["/bin/bash", "-l", "-c"] 
RUN apt-get update && apt-get install -y curl git
COPY --from=build-anaconda /opt/anaconda3 /opt/anaconda3
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y dpkg-dev gcc g++ binutils libx11-dev libxpm-dev libxft-dev libxext-dev libgsl-dev libfftw3-dev libpcre3-dev

WORKDIR /opt/
RUN curl -O https://root.cern/download/root_v6.22.02.source.tar.gz
RUN tar xf root_v6.22.02.source.tar.gz
RUN rm root_v6.22.02.source.tar.gz
WORKDIR /opt/root-6.22.02/build-root
RUN cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/opt/root -Dpython3=ON -Ddavix=OFF -Dminuit2=ON -Droofit=ON /opt/root-6.22.02
RUN make -j12
RUN make install
RUN rm -rf /opt/root-6.22.02

FROM nvidia/cudagl:10.1-devel-ubuntu18.04  as build-geant4
LABEL Nicholas Ruof <nickruof@uw.edu>

USER root
COPY chroma_env.sh /etc/profile.d/
SHELL ["/bin/bash", "-l", "-c"] 
RUN apt-get update && apt-get install -y curl git
COPY --from=build-anaconda /opt/anaconda3 /opt/anaconda3
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y dpkg-dev gcc g++ binutils libx11-dev libxpm-dev libxft-dev libxext-dev libgsl-dev libfftw3-dev libpcre3-dev
COPY --from=build-root /opt/root /opt/root
RUN apt-get update && apt-get install -y libxerces-c-dev libxmu-dev libxi-dev freeglut3-dev

WORKDIR /opt/
RUN curl -O https://geant4-data.web.cern.ch/geant4-data/releases/geant4.10.05.p01.tar.gz
RUN tar xf geant4.10.05.p01.tar.gz
RUN rm geant4.10.05.p01.tar.gz
WORKDIR /opt/geant4.10.05.p01/build-g4
RUN cmake -DCMAKE_BUILD_TYPE=Release -DGEANT4_INSTALL_DATA=ON -DCMAKE_INSTALL_PREFIX=/opt/geant4 -DGEANT4_USE_GDML=ON -DGEANT4_USE_OPENGL_X11=ON -DGEANT4_USE_XM=OFF /opt/geant4.10.05.p01
RUN make -j12
RUN make install
WORKDIR /opt/geant4.10.05.p01
COPY g4py.4.10.05.p01.patch ./
RUN git apply g4py.4.10.05.p01.patch
WORKDIR /opt/geant4.10.05.p01/build-g4py
RUN BOOST_ROOT=/opt/anaconda3 cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/opt/geant4 /opt/geant4.10.05.p01/environments/g4py
RUN make -j12 
RUN sed -i 's/install: preinstall/install:/' Makefile
RUN find . -name '*.cmake' -exec sed -i -n -E '/\.pyc|\.pyo/!p' {} \;
RUN sed -i -E 's/(.*G4LossTableManager.Instance.*)/#\1/' source/python3/__init__.py
RUN make install
RUN rm -rf /opt/geant4.10.05.p01


FROM nvidia/cudagl:10.1-devel-ubuntu18.04
LABEL Nicholas Ruof <nickruof@uw.edu>

USER root
COPY chroma_env.sh /etc/profile.d/
SHELL ["/bin/bash", "-l", "-c"] 
RUN apt-get update && apt-get install -y curl git
COPY --from=build-anaconda /opt/anaconda3 /opt/anaconda3
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y dpkg-dev gcc g++ binutils libx11-dev libxpm-dev libxft-dev libxext-dev libgsl-dev libfftw3-dev libpcre3-dev
COPY --from=build-root /opt/root /opt/root
RUN apt-get update && apt-get install -y libxerces-c-dev libxmu-dev libxi-dev freeglut3-dev
COPY --from=build-geant4 /opt/geant4 /opt/geant4
RUN apt-get update && apt-get install -y libgmp-dev libmpfr-dev libgmpxx4ldbl zip unzip patchelf
RUN pip install numpy matplotlib h5py jupyterlab ipympl jupyterlab_widgets pandas seaborn pyvista tinyDB virtualenvwrapper numba pythreejs panel ipygany ipyvtklink
RUN apt-get -y install wget npm
RUN cd /opt && wget https://nodejs.org/dist/v16.13.0/node-v16.13.0.tar.gz && tar -xvf node-v16.13.0.tar.gz && cd node-v16.13.0 && ./configure && make all -j8 && make install
RUN jupyter labextension install @arbennett/base16-nord && jupyter labextension install jupyterlab-spreadsheet \
    && jupyter labextension install jupyterlab-drawio
COPY overrides.json /opt/anaconda3/share/jupyter/lab/settings
RUN git clone --branch v0.3 https://github.com/PyMesh/PyMesh.git && \
    cd PyMesh && \
    git submodule update --init && \
    python setup.py build && \
    python setup.py install && \
    cd ..

CMD ["/bin/bash","-l"]
