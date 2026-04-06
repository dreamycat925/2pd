# Two-Point Discrimination — Docker

Notes for running this Streamlit app locally with Docker / Docker Compose.

## Standard Run

Run this in the project root:

```bash
docker compose up -d --build
```

Open:

```text
http://localhost:50000
```

Stop:

```bash
docker compose down
```

## Development Run

If you want to mount the local source into the container so edits are reflected more easily:

```bash
docker compose -f docker-compose.dev.yml up -d --build
```

Open:

```text
http://localhost:50000
```

Stop:

```bash
docker compose -f docker-compose.dev.yml down
```

## Current Port Mapping

`docker-compose.yml` and `docker-compose.dev.yml` use:

```text
127.0.0.1:50000:8501
```

- Inside the container: `8501`
- On the host: `50000`
- Because it is bound to `127.0.0.1`, it is not exposed to the LAN

## Notes

- The container starts `2pd_discrimination_streamlit_app.py`
- If the port is already in use, change the left-side port in the compose file
  - Example: `127.0.0.1:51000:8501`
- If updates do not seem to apply, run `docker compose down` and then `docker compose up -d --build` again
