import os


def run():

    for settings in [
        ("half_cheetah", "off_policy", "mean", "100000"),
        ("half_cheetah", "off_policy", "mean", "20000"),
    ]:
        env_name, prediction, policy_loss, buffer_capacity = settings
        for i in range(3):
            launch_command = (
                "python -m mprl.ui.start alg=sac_tr "
                "alg.policy_loss={} "
                "prediction={} "
                "buffer.capacity={} "
                "env={} run_id={}".format(
                    policy_loss,
                    prediction,
                    buffer_capacity,
                    env_name,
                    i,
                )
            )
            file_content = "#!/bin/bash\n" + launch_command
            file_name = "sac_tr_{}_{}_{}_{}_{}.sh".format(
                env_name, i, prediction, policy_loss, buffer_capacity
            )
            with open(file_name, "w") as text_file:
                text_file.write(file_content)
            os.system("chmod +x {}".format(file_name))
            os.system(
                "sbatch -p single -N 1 -t 72:00:00 --mem=4000 {}".format(file_name)
            )


if __name__ == "__main__":
    run()
