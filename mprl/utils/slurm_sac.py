import os


def run():

    for env_name in [
        "meta_original_reacher",
        # "meta_original_button_press",
        # "meta_original_window_open",
    ]:
        for alpha in [0.01, 0.1, 0.5, 1.0]:
            for i in range(3):
                launch_command = (
                    "python -m mprl.ui.run alg=sac "
                    "env={} run_id={} alg.hyper.alpha={}".format(
                        env_name,
                        i,
                        alpha,
                    )
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
