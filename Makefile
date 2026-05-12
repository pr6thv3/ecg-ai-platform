.PHONY: build up down logs test clean

# Default target when running just `make`
default: up

# Build all Docker containers
build:
	docker-compose build

# Start the entire system (builds if images are missing)
up:
	docker-compose up --build -d
	@echo "System is starting! Frontend will be at http://localhost:3000"

# Stop and remove all containers, networks, and volumes
down:
	docker-compose down

# Tail logs for all services (press Ctrl+C to exit)
logs:
	docker-compose logs -f

# Run tests in the running containers
test:
	@echo "Running Backend Tests..."
	docker-compose exec backend pytest tests/
	@echo "Running Frontend Tests..."
	docker-compose exec frontend npm test

# Clean up dangling images and stopped containers
clean: down
	docker system prune -f
