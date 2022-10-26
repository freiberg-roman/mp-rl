import os


def run():

    for env_name in [
        "half_cheetah",
    ]:
        for prediction in ["off_policy"]:
            for type in ["mean_performance"]:
                for alpha in ["0.0", "0.0001", "0.005", "0.01", "0.5"]:
                    for i in range(5):
                        launch_command = (
                            "python -m mprl.ui.run alg=sac_tr "
                            "alg.hyper.policy_loss={} "
                            "prediction={} "
                            "alg.hyper.alpha={} "
                            "env={} run_id={}".format(
                                type,
                                prediction,
                                alpha,
                                env_name,
                                i,
                            )
                        )
                        file_content = "#!/bin/bash\n" + launch_command
                        file_name = "sac_mixed_mp_{}_{}_{}_{}.sh".format(
                            env_name, i, prediction, alpha
                        )
                        with open(file_name, "w") as text_file:
                            text_file.write(file_content)
                        os.system("chmod +x {}".format(file_name))
                        os.system(
                            "sbatch -p single -N 1 -t 72:00:00 --mem=8000 {}".format(
                                file_name
                            )
                        )


if __name__ == "__main__":
    run()