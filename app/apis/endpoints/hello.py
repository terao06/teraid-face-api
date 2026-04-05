from fastapi import FastAPI


hello_app = FastAPI()


@hello_app.get("/")
def hello_world() -> dict[str, str]:
    return {"message": "Hello, World!"}
