import os


def run():
    for use_batch_norm in ["True"]:
        for var in ["0.1", "0.25", "0.5", "0.75", "1.0", "1.5"]:
            for i in range(3):
                launch_command = (
                    "python -m mprl.ui.train_model "
                    "prediction=moe "
                    "prediction.variance={} "
                    "prediction.use_batch_normalization={} "
                    "run_id={}".format(
                        var,
                        use_batch_norm,
                        i,
                    )
                )
                file_content = "#!/bin/bash\n" + launch_command
                file_name = "model_{}_{}_{}.sh".format(var, use_batch_norm, i)
                with open(file_name, "w") as text_file:
                    text_file.write(file_content)
                os.system("chmod +x {}".format(file_name))
                os.system(
                    "sbatch -p single -N 1 -t 72:00:00 --mem=4000 {}".format(file_name)
                )


if __name__ == "__main__":
    run()
