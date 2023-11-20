import paramiko
from scp import SCPClient
from dotenv import load_dotenv
import os

FILENAMES = []


def fetch_files(
    remote_ip, username, password, remote_directory, local_directory, filenames
):
    """
    Fetches files from a remote server using SCP.
    :param remote_ip: IP address of the remote server
    :param username: Username for the remote server
    :param password: Password for the remote server
    :param remote_directory: Directory on the remote server where the files are located
    :param local_directory: Directory where the files should be saved
    :param filenames: List of filenames to fetch
    :return: None
    """
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        # Connect to the remote server
        ssh.connect(remote_ip, username=username, password=password)

        # SCPCLient takes a paramiko transport as an argument
        with SCPClient(ssh.get_transport()) as scp:
            for filename in filenames:
                remote_path = f"{remote_directory}/sentence{filename.split('_')[1][1:]}/3Dmesh/{filename}.obj"
                local_path = f"{local_directory}/{filename}.obj"
                scp.get(remote_path, local_path)

        print("Files fetched successfully.")

    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        ssh.close()


load_dotenv()

remote_ip = os.getenv("remote_ip")
username = os.getenv("username")
password = os.getenv("password")
remote_path = os.getenv("remote_path")
download_path = os.getenv("download_path")

fetch_files(remote_ip, username, password, remote_path, download_path, FILENAMES)
