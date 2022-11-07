from mprl.models.sac_mp.tr.agent import SACTRL
from mprl.utils import SequenceRB
from mprl.utils.ds_helper import compare_networks


class PlannerStub:
    def reset_planner(self):
        pass


def test_sac_load_store():
    sac_agent = SACTRL(
        gamma=0.99,
        tau=0.005,
        alpha=0.2,
        automatic_entropy_tuning=True,
        alpha_q=0.0,
        num_steps=1,
        lr=0.0003,
        batch_size=256,
        state_dim=8,
        action_dim=2,
        num_basis=4,
        network_width=256,
        network_depth=2,
        action_scale=1.0,
        planner_act=PlannerStub(),
        planner_eval=PlannerStub(),
        planner_update=PlannerStub(),
        planner_imp_sampling=PlannerStub(),
        ctrl=None,
        buffer=SequenceRB(
            capacity=1000000,
            state_dim=8,
            action_dim=2,
            sim_qp_dim=1,
            sim_qv_dim=1,
            weight_mean_dim=3,
            weight_std_dim=3,
            minimum_sequence_length=1,
        ),
        decompose_fn=None,
        model=None,
        kl_loss_scale=1.0,
    )

    # Store the agent
    sac_agent.store(sac_agent.store_under("./"))

    sac_agent_two = SACTRL(
        gamma=0.99,
        tau=0.005,
        alpha=0.2,
        automatic_entropy_tuning=True,
        alpha_q=0.0,
        num_steps=1,
        lr=0.0003,
        batch_size=256,
        state_dim=8,
        action_dim=2,
        num_basis=4,
        network_width=256,
        network_depth=2,
        action_scale=1.0,
        planner_act=PlannerStub(),
        planner_eval=PlannerStub(),
        planner_update=PlannerStub(),
        planner_imp_sampling=PlannerStub(),
        ctrl=None,
        buffer=SequenceRB(
            capacity=1000000,
            state_dim=8,
            action_dim=2,
            sim_qp_dim=1,
            sim_qv_dim=1,
            weight_mean_dim=3,
            weight_std_dim=3,
            minimum_sequence_length=1,
        ),
        decompose_fn=None,
        model=None,
        kl_loss_scale=1.0,
    )
    sac_agent_two.load(sac_agent.store_under("./"))

    assert compare_networks(sac_agent, sac_agent_two)

    import shutil

    shutil.rmtree(sac_agent.store_under("./"))
