import paramiko
from dotenv import load_dotenv
import os


def find_json_files(remote_ip, username, password, remote_path, local_file):
    # Initialize SSH client
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        # Connect to the remote server
        ssh.connect(remote_ip, username=username, password=password)

        # Command to find all JSON files
        command = f"find {remote_path} -type f -name '*.json'"

        # Execute the command
        stdin, stdout, stderr = ssh.exec_command(command)

        # Read the output and write it to a local file
        with open(local_file, "w") as file:
            for line in stdout:
                file.write(line)

        print(f"JSON file list saved to {local_file}")

    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        ssh.close()


local_file = "json_files_list.txt"

load_dotenv()
remote_ip = os.getenv("remote_ip")
username = os.getenv("username")
password = os.getenv("password")
remote_path = os.getenv("remote_path")

# Execute the function
find_json_files(remote_ip, username, password, remote_path, local_file)
