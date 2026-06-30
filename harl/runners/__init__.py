"""Runner registry."""
from harl.runners.on_policy_ha_runner import OnPolicyHARunner
from harl.runners.on_policy_ma_runner import OnPolicyMARunner
from harl.runners.on_policy_ippo_runner import OnPolicyIPPORunner
from harl.runners.on_policy_ar_runner import OnPolicyARRunner
from harl.runners.on_policy_stage_runner import OnPolicyStageRunner
from harl.runners.off_policy_ha_runner import OffPolicyHARunner
from harl.runners.off_policy_ma_runner import OffPolicyMARunner
from harl.runners.off_policy_stage_runner import OffPolicyStageRunner

RUNNER_REGISTRY = {
    "happo": OnPolicyHARunner,
    "hatrpo": OnPolicyHARunner,
    "haa2c": OnPolicyHARunner,
    "haddpg": OffPolicyHARunner,
    "hatd3": OffPolicyHARunner,
    "hasac": OffPolicyHARunner,
    "had3qn": OffPolicyHARunner,
    "maddpg": OffPolicyMARunner,
    "stage_maddpg": OffPolicyStageRunner,
    "matd3": OffPolicyMARunner,
    "mappo": OnPolicyMARunner,
    "ma_mappo": OnPolicyMARunner,
    "ar_mappo": OnPolicyARRunner,
    "stage_mappo": OnPolicyStageRunner,
    "ippo": OnPolicyIPPORunner,
    "r_mappo": OnPolicyMARunner,
    "mappo_potential": OnPolicyMARunner,
    "happo_potential": OnPolicyHARunner,
    "hatrpo_potential": OnPolicyHARunner,
}
