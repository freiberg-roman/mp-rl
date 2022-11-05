import os


def run():

    for env_name in [
        "meta_window_open",
    ]:
        for i in range(10):
            launch_command = (
                "python -m mprl.ui.start alg=sac "
                "env={} run_id={}".format(
                    env_name,
                    i,
                )
            )
            file_content = "#!/bin/bash\n" + launch_command
            file_name = "sac_{}_{}.sh".format(env_name, i)
            with open(file_name, "w") as text_file:
                text_file.write(file_content)
            os.system("chmod +x {}".format(file_name))
            os.system(
                "sbatch -p single -N 1 -t 72:00:00 --mem=4000 {}".format(file_name)
            )


if __name__ == "__main__":
    run()
