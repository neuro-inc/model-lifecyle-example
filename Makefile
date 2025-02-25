.PHONY: build

IMAGE_NAME = model_lifecycle_example
IMAGE_TAG = latest

.PHONY: build-train
build-train:
	docker build -t $(IMAGE_NAME)_train:$(IMAGE_TAG) -f ./scripts/Dockerfile ./scripts

.PHONY: build-serve
build-serve:
	docker build -t $(IMAGE_NAME)_serve:$(IMAGE_TAG) -f ./scripts/Dockerfile.server ./scripts

.PHONY: build
build: build-train build-serve

.PHONY: push-apolo
push-apolo:	
	apolo push $(IMAGE_NAME)_train:$(IMAGE_TAG)  image:$(IMAGE_NAME)_train:$(IMAGE_TAG)
	apolo push $(IMAGE_NAME)_serve:$(IMAGE_TAG) image:$(IMAGE_NAME)_serve:$(IMAGE_TAG)