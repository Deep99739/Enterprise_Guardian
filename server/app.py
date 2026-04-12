"""
Enterprise Guardian — FastAPI application entry point.

Bootstraps the OpenEnv HTTP server with the environment class,
action/observation schemas, and concurrency configuration.
"""

try:
    from openenv.core.env_server.http_server import create_app
    from ..models import EnterpriseGuardianAction, EnterpriseGuardianObservation
    from .environment import EnterpriseGuardianEnvironment
except ImportError:
    from openenv.core.env_server.http_server import create_app
    from models import EnterpriseGuardianAction, EnterpriseGuardianObservation
    from server.environment import EnterpriseGuardianEnvironment

app = create_app(
    EnterpriseGuardianEnvironment,
    EnterpriseGuardianAction,
    EnterpriseGuardianObservation,
    env_name="enterprise_guardian",
    max_concurrent_envs=1,
)


def main(host: str = "0.0.0.0", port: int = 7860):
    """Entry point for `uv run server` and direct execution."""
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
