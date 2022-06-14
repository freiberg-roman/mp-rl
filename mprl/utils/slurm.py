import os

for env_name in ["half_cheetah", "ant", "hopper"]:
    for i in range(10):
        launch_command = "python -m mprl.main.main env={} run={}".format(env_name, i)
        file_content = "#!/bin/bash \n" + launch_command
        file_name = "{}_{}".format(env_name, i)
        with open(file_name + ".sh", "w") as text_file:
            text_file.write(file_content)
        os.system("chmod +x {}".format(file_name))
        os.system("sbatch -p single -N 1 -t 48:00:00 --mem=6000 {}".format(file_name))
