import os


def run():

    for env_name in [
        "half_cheetah",
    ]:
        for tau, num_steps in zip([0.5, 1.0, 2.5], [10, 20, 50]):
            for prediction in ["off_policy", "ground_truth"]:
                for i in range(3):
                    launch_command = (
                        "python -m mprl.ui.run alg=sac_mixed_mp prediction={} "
                        "env={} run_id={} alg.hyper.num_steps={} alg.mp.tau={}".format(
                            prediction,
                            env_name,
                            i,
                            num_steps,
                            tau,
                        )
                    )
                    file_content = "#!/bin/bash\n" + launch_command
                    file_name = "sac_mixed_mp_{}_{}_{}_{}_{}.sh".format(
                        env_name, i, prediction, tau, num_steps
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
