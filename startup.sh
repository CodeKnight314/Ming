# Context Redis (URL cache, etc.)
if docker ps -a --format '{{.Names}}' | grep -q '^redis$'; then
  if docker ps --format '{{.Names}}' | grep -q '^redis$'; then
    echo "Context Redis is already running."
  else
    echo "Starting existing context Redis container..."
    docker start redis
  fi
else
  echo "Creating and starting context Redis container..."
  docker run -d -p 6379:6379 --name redis redis:latest
fi

# Queries Redis (search query history for generation/think context)
if docker ps -a --format '{{.Names}}' | grep -q '^redis-queries$'; then
  if docker ps --format '{{.Names}}' | grep -q '^redis-queries$'; then
    echo "Queries Redis is already running."
  else
    echo "Starting existing queries Redis container..."
    docker start redis-queries
  fi
else
  echo "Creating and starting queries Redis container..."
  docker run -d -p 6380:6379 --name redis-queries redis:latest
fi