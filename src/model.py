from pydantic import BaseModel

class QueryFrame(BaseModel):
    intent: str
    camera: str | None = None
    date_expression: str | None = None
    start_time: str | None = None
    end_time: str | None = None
    weekday: str | None = None
    
class ResolvedQuery(BaseModel):