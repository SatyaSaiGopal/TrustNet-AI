"""Run the TrustNet AI API and frontend server."""

import uvicorn

from trustnet.api.app import app


def main() -> None:
    print("Starting TrustNet AI server on http://127.0.0.1:8000 ...")
    uvicorn.run("trustnet.api.app:app", host="127.0.0.1", port=8000, reload=True)


if __name__ == "__main__":
    main()
