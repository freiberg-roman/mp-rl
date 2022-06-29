import os


def run():
    for alg in ["sac_stepwise_mp", "sac_mp"]:
        for env_name in ["half_cheetah", "ant", "hopper"]:
            for i in range(5):
                launch_command = (
                    "python -m mprl.main.run algorithm={} env={} run_id={}".format(
                        alg, env_name, i
                    )
                )
                file_content = "#!/bin/bash \n" + launch_command
                file_name = "{}_{}_{}.sh".format(alg, env_name, i)
                with open(file_name, "w") as text_file:
                    text_file.write(file_content)
                os.system("chmod +x {}".format(file_name))
                os.system(
                    "sbatch -p single -N 1 -t 72:00:00 --mem=6000 {}".format(file_name)
                )


if __name__ == "__main__":
    run()
