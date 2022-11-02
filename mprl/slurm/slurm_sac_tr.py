import os


def run():
    for env_name in [
        "half_cheetah",
    ]:
        for eps_mean in ["0.01", "0.001"]:
            for eps_cov in ["0.001", "0.0001", "0.00001"]:
                for policy_loss in ["off_policy", "moe"]:
                    for i in range(3):
                        launch_command = (
                            "python -m mprl.ui.run alg=sac_tr "
                            "alg.hyper.policy_loss={} "
                            "prediction=mean_performance "
                            "alg.hyper.mean_bound={} "
                            "alg.hyper.cov_bound={} "
                            "env={} run_id={}".format(
                                policy_loss,
                                eps_mean,
                                eps_cov,
                                env_name,
                                i,
                            )
                        )
                        file_content = "#!/bin/bash\n" + launch_command
                        file_name = "sac_tr_{}_{}_{}_{}.sh".format(
                            eps_mean, eps_cov, env_name, i
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
