from mprl.models.sac import SAC
from mprl.utils import RandomRB
from mprl.utils.ds_helper import compare_networks


def test_sac_load_store():
    sac_agent = SAC(
        gamma=0.99,
        tau=0.005,
        alpha=0.2,
        automatic_entropy_tuning=True,
        lr=0.0003,
        batch_size=256,
        state_dim=8,
        action_dim=2,
        network_width=256,
        network_depth=2,
        buffer=RandomRB(1000000, 8, 2, 1, 1),
    )

    # Store the agent
    sac_agent.store("./" + sac_agent.store_under())

    sac_agent_two = SAC(
        gamma=0.99,
        tau=0.005,
        alpha=0.2,
        automatic_entropy_tuning=True,
        lr=0.0003,
        batch_size=256,
        state_dim=8,
        action_dim=2,
        network_width=256,
        network_depth=2,
        buffer=RandomRB(1000000, 8, 2, 1, 1),
    )
    sac_agent_two.load("./" + sac_agent.store_under())

    assert compare_networks(sac_agent.critic, sac_agent_two.critic)
    assert compare_networks(sac_agent.critic_target, sac_agent_two.critic_target)
    assert compare_networks(sac_agent.policy, sac_agent_two.policy)

    import shutil

    shutil.rmtree("./" + sac_agent.store_under())
