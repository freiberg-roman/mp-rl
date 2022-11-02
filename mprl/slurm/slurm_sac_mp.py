import os


def run():

    for env_name in [
        "half_cheetah",
    ]:
        for alpha in [0.5, 1.0]:
            for i in range(3):
                launch_command = (
                    "python -m mprl.ui.run alg=sac_mp "
                    "env={} run_id={} alg.hyper.alpha={}".format(
                        env_name,
                        i,
                        alpha,
                    )
                )
                file_content = "#!/bin/bash\n" + launch_command
                file_name = "sac_mp_{}_{}_{}.sh".format(env_name, i, alpha)
                with open(file_name, "w") as text_file:
                    text_file.write(file_content)
                os.system("chmod +x {}".format(file_name))
                os.system(
                    "sbatch -p single -N 1 -t 72:00:00 --mem=8000 {}".format(file_name)
                )


if __name__ == "__main__":
    run()
