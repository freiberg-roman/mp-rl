import os


def run():
    for prediction in ["moe", "ground_truth", "off_policy"]:
        for weighting in ["likelihood"]:
            for env_name in ["reacher", "half_cheetah", "ant"]:
                for i in range(5):
                    launch_command = (
                        "python -m mprl.main.run algorithm=sac_stepwise_mp "
                        "algorithm/prediction={} "
                        "algorithm.reward_weighting={} "
                        "env={} run_id={}".format(
                            prediction,
                            weighting,
                            env_name,
                            prediction + "_" + weighting + "_" + str(i) + "_v2",
                        )
                    )
                    file_content = "#!/bin/bash\n" + launch_command
                    file_name = "{}_{}_{}_{}.sh".format(
                        prediction, weighting, env_name, i
                    )
                    with open(file_name, "w") as text_file:
                        text_file.write(file_content)
                    os.system("chmod +x {}".format(file_name))
                    os.system(
                        "sbatch -p single -N 1 -t 72:00:00 --mem=6000 {}".format(
                            file_name
                        )
                    )


if __name__ == "__main__":
    run()
