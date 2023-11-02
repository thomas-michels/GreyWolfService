from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware
from app.api.routers import model_router


def create_app() -> FastAPI:
    app = FastAPI(
        title="GreyWolf Services"
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins="*",
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(model_router)

    return app
