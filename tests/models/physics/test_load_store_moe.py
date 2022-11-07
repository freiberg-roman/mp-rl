from mprl.models.physics.moe import MixtureOfExperts
from mprl.utils.ds_helper import compare_networks


def test_load_store_moe():
    moe = MixtureOfExperts(
        state_dim_in=3,
        state_dim_out=3,
        action_dim=2,
        num_experts=5,
        network_width=6,
        variance=7.0,
        lr=8e-4,
    )
    moe_two = MixtureOfExperts(
        state_dim_in=3,
        state_dim_out=3,
        action_dim=2,
        num_experts=5,
        network_width=6,
        variance=7.0,
        lr=8e-4,
    )

    moe.store(moe.store_under("./"))
    moe_two.load(moe_two.store_under("./"))
    assert compare_networks(moe, moe_two)

    import shutil

    shutil.rmtree(moe.store_under("./"))
