include ./.env

build:
	docker build -t grey-wolf-service --no-cache .

run:
	docker run --env-file .env --network ${DEV_CONTAINER_NETWORK} -p ${APPLICATION_PORT}:8000 --name grey-wolf-service -d grey-wolf-service
