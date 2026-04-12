from openenv_core.env_server import create_app

from ..models import EnterpriseGuardianAction, EnterpriseGuardianObservation
from .environment import EnterpriseGuardianEnvironment

# We instantiate the environment here
env_instance = EnterpriseGuardianEnvironment()

# create_app takes the environment instance and the expected Action/Observation classes
app = create_app(
    env=env_instance,
    action_cls=EnterpriseGuardianAction,
    observation_cls=EnterpriseGuardianObservation,
    env_name="enterprise_guardian"
)
