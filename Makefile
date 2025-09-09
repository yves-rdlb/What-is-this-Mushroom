## Clean, consolidated Makefile

#======================#
# Install, clean, test #
#======================#

install_requirements:
	@python -m pip install --upgrade pip
	@python -m pip install -r requirements.txt

install:
	@pip install . -U

clean:
	@rm -f */version.txt
	@rm -f .coverage
	@rm -fr */__pycache__ */*.pyc __pycache__
	@rm -fr build dist
	@rm -fr proj-*.dist-info
	@rm -fr proj.egg-info

#======================#
#          UI          #
#======================#

streamlit:
	python -m streamlit run UI/app_v3.py

#======================#
#          API         #
#======================#

run_api:
	uvicorn MUSHROOM.api.fast:app --reload --port 8000

api_health:
	@echo "GET /" && curl -s http://127.0.0.1:8000/ || true
	@echo "\nPOST /predict (python)" && python scripts/health_check_api.py --url http://127.0.0.1:8000/predict || true

run_vit_api	:
	uvicorn MUSHROOM.api.VIT_API:app --reload --port 8000

#======================#
#          GCP         #
#======================#

gcloud-set-project:
	gcloud config set project $(GCP_PROJECT)



#======================#
#         Docker       #
#======================#

# Local images - using local computer's architecture
# i.e. linux/amd64 for Windows / Linux / Apple with Intel chip
#      linux/arm64 for Apple with Apple Silicon (M1 / M2 chip)

docker_build_local:
	docker build --tag=$(DOCKER_IMAGE_NAME):local .

docker_run_local:
	docker run \
		-e PORT=8000 -p $(DOCKER_LOCAL_PORT):8000 \
		--env-file .env \
		$(DOCKER_IMAGE_NAME):local

docker_run_local_interactively:
	docker run -it \
		-e PORT=8000 -p $(DOCKER_LOCAL_PORT):8000 \
		--env-file .env \
		$(DOCKER_IMAGE_NAME):local \
		bash

# Cloud images - using architecture compatible with cloud, i.e. linux/amd64

DOCKER_IMAGE_PATH := $(GCP_REGION)-docker.pkg.dev/$(GCP_PROJECT)/$(DOCKER_REPO_NAME)/$(DOCKER_IMAGE_NAME)

docker_show_image_path:
	@echo $(DOCKER_IMAGE_PATH)

docker_build:
	docker build \
		--platform linux/amd64 \
		-t $(DOCKER_IMAGE_PATH):prod .

docker_build_alternative:
	docker buildx build --load \
		--platform linux/amd64 \
		-t $(DOCKER_IMAGE_PATH):prod .

docker_run:
	docker run \
		--platform linux/amd64 \
		-e PORT=8000 -p $(DOCKER_LOCAL_PORT):8000 \
		--env-file .env \
		$(DOCKER_IMAGE_PATH):prod

docker_run_interactively:
	docker run -it \
		--platform linux/amd64 \
		-e PORT=8000 -p $(DOCKER_LOCAL_PORT):8000 \
		--env-file .env \
		$(DOCKER_IMAGE_PATH):prod \
		bash

docker_allow:
	gcloud auth configure-docker $(GCP_REGION)-docker.pkg.dev

docker_create_repo:
	gcloud artifacts repositories create $(DOCKER_REPO_NAME) \
		--repository-format=docker \
		--location=$(GCP_REGION) \
		--description="Repository for storing docker images"

docker_push:
	docker push $(DOCKER_IMAGE_PATH):prod

docker_deploy:
	gcloud run deploy \
		--image $(DOCKER_IMAGE_PATH):prod \
		--memory $(GAR_MEMORY) \
		--region $(GCP_REGION) \
		--env-vars-file .env.yaml
