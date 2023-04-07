ENV_FILE ?= .env

build:
	docker-compose -p fastapi -f docker-compose.yml build

serve:
	docker-compose -p fastapi -f docker-compose.yml --env-file ./${ENV_FILE} up
