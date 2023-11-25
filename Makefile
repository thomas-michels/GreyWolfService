include ./.env

build:
	docker build -t grey-wolf-service --no-cache .

build_consumer:
	docker build --file ./Dockerfile.consumer -t grey-wolf-service-consumer --no-cache .

run:
	docker run --env-file .env --network ${DEV_CONTAINER_NETWORK} -p ${APPLICATION_PORT}:8000 --name grey-wolf-service -d grey-wolf-service

run_consumer:
	docker run --env-file .env --network ${DEV_CONTAINER_NETWORK} --name grey-wolf-service-consumer -d grey-wolf-service-consumer
