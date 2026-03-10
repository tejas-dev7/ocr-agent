# Qdrant Setup Guide

Qdrant is a vector database used by OCR Agent for storing document chunks and enabling semantic search.

## Option 1: Docker (Recommended)

### Quick start

```bash
# Pull and run Qdrant (data stored in Docker volume, not in project dir)
docker run -d \
  --name qdrant \
  -p 6333:6333 \
  -p 6334:6334 \
  -v qdrant_data:/qdrant/storage \
  qdrant/qdrant
```

- **REST API:** http://localhost:6333
- **Web dashboard:** http://localhost:6333/dashboard
- **gRPC:** port 6334

### Verify it's running

```bash
curl http://localhost:6333/healthz
# Should return: {"title":"qdrant - vector search engine","version":"..."}
```

---

## Option 2: Docker Compose

Create `docker-compose.yml` in your project:

```yaml
services:
  qdrant:
    image: qdrant/qdrant:latest
    container_name: qdrant
    restart: unless-stopped
    ports:
      - "6333:6333"
      - "6334:6334"
    volumes:
      - qdrant_data:/qdrant/storage

volumes:
  qdrant_data:
```

Run:

```bash
docker compose up -d
```

---

## Option 3: Native binary (no Docker)

1. Download from [Qdrant releases](https://github.com/qdrant/qdrant/releases)
2. Or use the install script:

```bash
# Linux/macOS
curl -sSfL https://raw.githubusercontent.com/qdrant/qdrant/master/scripts/install.sh | sh
```

3. Run:

```bash
./qdrant
# Listens on http://localhost:6333
```

---

## Connect OCR Agent to Qdrant

1. **Install Qdrant client:**
   ```bash
   pip install "ocr-agent[qdrant]"
   ```

2. **Set in `.env`:**
   ```bash
   OCR_STORAGE_BACKEND=qdrant
   OCR_QDRANT_URL=http://localhost:6333
   ```

3. **If Qdrant runs elsewhere:**
   ```bash
   OCR_QDRANT_URL=http://your-server:6333
   ```

---

## Useful commands

| Action | Command |
|--------|---------|
| Stop | `docker stop qdrant` |
| Start | `docker start qdrant` |
| View logs | `docker logs qdrant` |
| Remove | `docker rm -f qdrant` |
| Dashboard | Open http://localhost:6333/dashboard |

---

## Storage persistence

With the `-v` volume mount, data is stored in a Docker named volume (`qdrant_data`) or in Docker Compose's `qdrant_data` volume—not in your project directory. Restarting the container keeps your data.

**Using remote Qdrant?** Set `OCR_QDRANT_URL` to your cloud or server URL. No local storage needed.
