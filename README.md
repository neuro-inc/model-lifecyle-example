# model-lifecycle-example




## Quick Start

Sign up at [apolo](https://console.apolo.us) and setup your local machine according to [instructions](https://docs.apolo.us/).

Then run:

```shell
# make sure you have apolo cli installed
pip install -U pipx
pipx install apolo-all

# login to apolo, you may need to add the url to your control plane like `apolo login https://api.apolo.scottdata.ai/api/v1`
apolo login
# you may need to create a disk for mlflow if you haven't already
apolo disk create 2GB --name global-mlflow-db

# start mlflow instance. It should open a url where you can follow the experiments' metrics and parameters
apolo-flow run mlflow
# download data, you can also copy data from a url or bucket directly to apolo storage with apolo-extras cli
mkdir -p data && curl https://download.pytorch.org/tutorial/data.zip -o data/data.zip && unzip data/data.zip && rm data/data.zip

# build the training image
apolo-flow build train

# runs a training job and logs model metadata to mlflow
apolo-flow run train

# starts a serving job with fastapi. You can access the url /docs to see a Swagger documentation and you can even try the model from the web browser
apolo-flow run serve
```

See [Help.md](HELP.md) for the detailed flow template reference.
# model-lifecyle-example
