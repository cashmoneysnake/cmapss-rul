"""Request and response shapes (validated automatically by FastAPI/pydantic)."""
import math

from pydantic import BaseModel, Field, field_validator

N_VALUES = 24   # 3 operating settings + 21 sensors


class PredictRequest(BaseModel):
    cycles: list[list[float]] = Field(
        min_length=1, max_length=1000,
        description="One engine's recent cycles, oldest first. Each cycle is 24 numbers: "
                    "setting_1..3 then sensor s_1..s_21 (see GET /model-info).")

    @field_validator("cycles")
    @classmethod
    def check_rows(cls, rows):
        for i, row in enumerate(rows):
            if len(row) != N_VALUES:
                raise ValueError(f"cycle {i} has {len(row)} values, expected {N_VALUES}")
            if not all(math.isfinite(v) for v in row):
                raise ValueError(f"cycle {i} contains a non-finite value")
        return rows


class PredictResponse(BaseModel):
    rul: float = Field(description="Predicted remaining useful life in cycles (0 to the training cap)")
    cycles_used: int
    padded: bool = Field(description="True if fewer cycles than the model window were sent")
    operating_condition: int = Field(description="Operating-condition cluster of the latest cycle")
