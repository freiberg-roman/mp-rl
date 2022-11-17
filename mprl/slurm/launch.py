import argparse
import os

import tomli

parser = argparse.ArgumentParser(
    prog="Launch",
    description="Launches jobs on the cluster",
)

parser.add_argument("filename")


def main():
    args = parser.parse_args()
    with open(args.filename, "rb") as f:
        config = tomli.load(f)

    for (key, val) in config["run"].items():
        num_jobs = val["jobs"]
        algorithm = val["algorithm"]
        env = val["env"]
        project_name = val["project_name"]
        prediction = val.get("prediction", "off_policy")

        for i in range(num_jobs):
            file_content = "python -m mprl.ui.start alg={} env={} run_id={} project_name={} prediction={}".format(
                algorithm, env, i, project_name, prediction
            )
            for (setting, setting_val_val) in val.items():
                if isinstance(setting_val_val, dict):
                    for (setting_key, setting_val) in setting_val_val.items():
                        file_content += " {}.{}={}".format(
                            setting, setting_key, setting_val
                        )

            file_content = "#!/bin/bash\n" + file_content
            file_name = "{}_{}.sh".format(key, i)
            with open(file_name, "w") as text_file:
                text_file.write(file_content)
            os.system("chmod +x {}".format(file_name))
            os.system(
                "sbatch -p single -N 1 -t {}:00:00 --mem=4000 {}".format(
                    val["cluster_hours"], file_name
                )
            )


if __name__ == "__main__":
    main()
