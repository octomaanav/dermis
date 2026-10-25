
from uagents import Agent, Context
from messages import SensorRequest, SensorResult
from sensor_agent import sensor

orchestrator = Agent(
    name="dermis_orchestrator",
    seed="dermis-orchestrator-local-demo-secret",
    port=8000,
    endpoint=["http://localhost:8000/submit"]
)

@orchestrator.on_interval(period=10.0)
async def request_analysis(ctx: Context):
    result, status = await ctx.send_and_receive(
        sensor.address,
        SensorRequest(session_id="session_001"),
        response_type=SensorResult,
        timeout=8
    )

    if isinstance(result, SensorResult):
        ctx.logger.info(f"Analysis: {result}")
    else:
        ctx.logger.error(f"Request failed: {status}")

if __name__ == "__main__":
    orchestrator.run()

