import os


def run():

    for env_name in [
        "half_cheetah",
        "meta_reach",
        "meta_window_open",
        "meta_button_press",
    ]:
        for prediction in ["off_policy"]:
            for type in ["mean_performance"]:
                for alpha_q in ["0.025", "0.0"]:
                    for i in range(5):
                        launch_command = (
                            "python -m mprl.ui.run alg=sac_mixed_mp "
                            "alg.hyper.policy_loss={} "
                            "prediction={} "
                            "alg.hyper.alpha_q={} "
                            "env={} run_id={}".format(
                                type,
                                prediction,
                                alpha_q,
                                env_name,
                                i,
                            )
                        )
                        file_content = "#!/bin/bash\n" + launch_command
                        file_name = "sac_mixed_mp_{}_{}_{}_{}.sh".format(
                            env_name, i, prediction, alpha_q
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
