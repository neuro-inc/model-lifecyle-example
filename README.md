# model-lifecycle-example




## Quick Start

Sign up at [apolo](https://console.apolo.us) and setup your local machine according to [instructions](https://docs.apolo.us/).

Then run:

```shell
pip install -U pipx
pipx install apolo-all
apolo login
apolo-flow build train
apolo-flow run mlflow
# copy data
curl https://download.pytorch.org/tutorial/data.zip -o data/data.zip && unzip data/data.zip && rm data/data.zip
apolo-flow run train
apolo-flow run serve
```

See [Help.md](HELP.md) for the detailed flow template reference.
# model-lifecyle-example
