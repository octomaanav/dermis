from uagents import Model

class SensorRequest(Model):
    session_id: str

class SensorResult(Model):
    session_id: str
    status: str
    sample_count: int
