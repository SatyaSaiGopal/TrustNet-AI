"""Run the TrustNet AI API and frontend server."""

import os
import uvicorn


def main() -> None:
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))
    reload = os.environ.get("RELOAD", "false").lower() in ("true", "1", "yes")

    print(f"Starting TrustNet AI server on http://{host}:{port} ...")
    uvicorn.run("trustnet.api.app:app", host=host, port=port, reload=reload)


if __name__ == "__main__":
    main()

