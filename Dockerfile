# Docker file for Titanic_Predictive_Data_Analysis
# Patrick Tung, Sylvia Lee (Dec 05, 2018)

# Description: This Makefile can be run to create our automatic
#							 data analysis pipeline.

# Usage:
#   To build the docker image: docker build --tag titanic_predictive_analysis:0.1 .
#		To create the report: docker run --rm -e PASSWORD=test -v <ABSOLUTE PATH OF REPO>:/home/titanic_predictive_analysis patricktung/sylvia_patrick_titanic_survival_ml make -C '/home/titanic_predictive_analysis' all
#		To get a clean start: docker run --rm -e PASSWORD=test -v <ABSOLUTE PATH OF REPO>:/home/titanic_predictive_analysis patricktung/sylvia_patrick_titanic_survival_ml make -C '/home/titanic_predictive_analysis' clean

# Use rocker/tidyverse as the base image
FROM rocker/tidyverse

# Install R packages
RUN Rscript -e "install.packages('here')"
RUN Rscript -e "install.packages('tinytex')"
RUN Rscript -e "tinytex::install_tinytex()"

# Install python 3
RUN apt-get update \
  && apt-get install -y python3-pip python3-dev \
  && cd /usr/local/bin \
  && ln -s /usr/bin/python3 python \
  && pip3 install --upgrade pip

# Get python package dependencies
RUN apt-get install -y python3-tk

# Install python packages
RUN pip3 install numpy
RUN pip3 install pandas
RUN pip3 install scikit-learn
RUN pip3 install seaborn
RUN apt-get install -y graphviz && pip install graphviz
RUN apt-get update && \
    pip3 install matplotlib && \
    rm -rf /var/lib/apt/lists/*
