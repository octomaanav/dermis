
from uagents import Agent, Context
from messages import SensorRequest, SensorResult

sensor = Agent(
    name="dermis_sensor",
    seed="dermis-sensor-local-demo-secret",
    port=8001,
    endpoint=["http://localhost:8001/submit"]
)

@sensor.on_event("startup")
async def startup(ctx: Context):
    ctx.logger.info(f"Sensor address: {sensor.address}")

@sensor.on_message(model=SensorRequest)
async def analyze(ctx: Context, sender: str, msg: SensorRequest):
    # TODO: Replace with real Dermis data analysis
    result = SensorResult(
        session_id=msg.session_id,
        status="demo_data_only",
        sample_count=100
    )

    await ctx.send(sender, result)

if __name__ == "__main__":
    sensor.run()

