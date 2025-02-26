# model-lifecycle-example




## Quick Start

Sign up at [apolo](https://console.apolo.us) and setup your local machine according to [instructions](https://docs.apolo.us/).

Then run:

```shell
pip install -U pipx
pipx install apolo-all
apolo login
# you may need to create a disk for mlflow if you haven't already
apolo disk create 2GB --name global-mlflow-db
apolo-flow run mlflow
# copy data
mkdir -p data && curl https://download.pytorch.org/tutorial/data.zip -o data/data.zip && unzip data/data.zip && rm data/data.zip
apolo-flow build train

apolo-flow run train
apolo-flow run serve
```

See [Help.md](HELP.md) for the detailed flow template reference.
# model-lifecyle-example
