# dogclassifier-web-backend

This is backend system for dog-classifier. You can find the notebook [here](https://github.com/ShubhamOjha/dog-classifer-udacity)

## Setup

* Setup the conda environment from `environment.yaml`.
   `conda install -f environment.yaml`
* Run the server using gunicorn : `nohup gunicorn backend.wsgi > gunicorn.out 2<&1 &`
