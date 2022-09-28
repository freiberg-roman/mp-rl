import os


def run():

    for env_name in [
        "half_cheetah",
        "meta_pd_reacher",
        "meta_pos_reacher",
    ]:
        for i in range(5):
            launch_command = "python -m mprl.ui.run alg=sac " "env={} run_id={}".format(
                env_name,
                i,
            )
            file_content = "#!/bin/bash\n" + launch_command
            file_name = "sac_{}_{}.sh".format(env_name, i)
            with open(file_name, "w") as text_file:
                text_file.write(file_content)
            os.system("chmod +x {}".format(file_name))
            os.system(
                "sbatch -p single -N 1 -t 72:00:00 --mem=8000 {}".format(file_name)
            )


if __name__ == "__main__":
    run()
