import modal

app = modal.App("research-harness-sandboxes")

sandbox_image = modal.Image.debian_slim(python_version="3.12")


@app.function(image=sandbox_image)
def image_healthcheck() -> dict[str, str]:
    import platform

    return {"status": "ok", "python": platform.python_version()}


@app.local_entrypoint()
def main() -> None:
    print(image_healthcheck.remote())
