import os


def run():

    for env_name in [
        "half_cheetah",
    ]:
        for prediction in ["off_policy", "moe"]:
            for type in ["mean_performance", "mean"]:
                for goal_scale in ["0.0", "2.0"]:
                    for i in range(3):
                        launch_command = (
                            "python -m mprl.ui.run alg=sac_mixed_mp "
                            "alg.hyper.policy_loss={} "
                            "prediction={} "
                            "alg.mp.mp_args.goals_scale={} "
                            "env={} run_id={}".format(
                                type,
                                prediction,
                                goal_scale,
                                env_name,
                                i,
                            )
                        )
                        file_content = "#!/bin/bash\n" + launch_command
                        file_name = "sac_mixed_mp_{}_{}_{}_{}.sh".format(
                            env_name, i, prediction, goal_scale
                        )
                        with open(file_name, "w") as text_file:
                            text_file.write(file_content)
                        os.system("chmod +x {}".format(file_name))
                        os.system(
                            "sbatch -p single -N 1 -t 72:00:00 --mem=4000 {}".format(
                                file_name
                            )
                        )


if __name__ == "__main__":
    run()
