if docker ps -a --format '{{.Names}}' | grep -q '^redis$'; then
  if docker ps --format '{{.Names}}' | grep -q '^redis$'; then
    echo "Redis is already running."
  else
    echo "Starting existing Redis container..."
    docker start redis
  fi
else
  echo "Creating and starting Redis container..."
  docker run -d -p 6379:6379 --name redis redis:latest
fi