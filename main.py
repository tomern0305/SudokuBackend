# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, conlist
from fastapi.middleware.cors import CORSMiddleware
from Solver import solve_vector

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

Vector81 = conlist(int, min_length=81, max_length=81)

class SolveRequest(BaseModel):
    vector: Vector81

class SolveResponse(BaseModel):
    vector: Vector81

@app.post("/solve", response_model=SolveResponse)
def solve(req: SolveRequest):
    try:
        solved = solve_vector(req.vector)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"vector": solved}
