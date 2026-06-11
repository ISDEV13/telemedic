import math
from fastapi import FastAPI
from pydantic import BaseModel, field_validator
from modules.calcul import carre
from prometheus_fastapi_instrumentator import Instrumentator

app = FastAPI()
Instrumentator().instrument(app).expose(app)


class NombreInput(BaseModel):
    nombre: float

    @field_validator('nombre')
    @classmethod
    def nombre_valide(cls, v):
        if math.isnan(v) or math.isinf(v):
            raise ValueError("Le nombre doit être un nombre fini")
        if v < -1_000_000 or v > 1_000_000:
            raise ValueError("Le nombre doit être compris entre -1 000 000 et 1 000 000")
        return v


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/carre/")
def get_carre(data: NombreInput):
    return {"nombre": data.nombre, "carre": carre(data.nombre)}
