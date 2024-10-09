
import os
import sys

import re
import shlex
import subprocess
import platform


def _split_preserve_dict(s):
    """
    Extension to "shlex.split" that doesn't break json objects.
    """

    # Temporarily replace the dictionary string
    # (carefully not doing it for cases like "${...}")
    dict_pattern = r'(?<!\$)(\{(?:[^{}]|\{[^{}]*\})*\})'
    dicts = re.findall(dict_pattern, s)
    placeholders = [f'__DICT{i}__' for i in range(len(dicts))]
    for placeholder, dict_str in zip(placeholders, dicts):
        s = s.replace(dict_str, placeholder, 1)
    
    # Perform the split
    tokens = shlex.split(s)
    
    # Restore the dictionary strings
    for placeholder, dict_str in zip(placeholders, dicts):
        tokens = [dict_str if token == placeholder else token
                  for token in tokens]
    
    return tokens

def _strip_ansi_escape_codes(text):
    # Remove ANSI escape codes from text
    ansi_escape = re.compile(r'\x1B\[.*?m')
    return ansi_escape.sub('', text)

def retrain_pipelines_local(
    command: str,
    env: os._Environ
) -> bool:

    last_stdout_line = ""

    command = _split_preserve_dict(command)

    # prepend script fullname to received list of params
    command = [
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "local_launcher.sh")
        ] + command

    if len(command) > 1:
        ############################################
        # parse named arguments from env variables #
        ############################################
        if "run" == command[2]:
            #print(env.keys())
            #print(os.environ.__dict__['_data'].keys())
            pattern = re.compile(r'^\$\{([^}]+)\}$')
            for index in range(4, len(command), 2):
                cmd_param_value = command[index]
                match = pattern.match(cmd_param_value)
                if match and match.group(1) in env.keys():
                    env_var_name = match.group(1)
                    command[index] = env[env_var_name]
        ############################################

    if platform.system() == 'Windows':
        import colorama
        from colorama import AnsiToWin32
        colorama.init(autoreset=True)
        output_stream = AnsiToWin32(sys.stdout).write

        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            shell=False, env=env, text=True
        )

        try:
            while True:
                output = process.stdout.read(1024)
                if not output:
                    break
                output_stream(output)
                sys.stdout.flush()
                lines = output.splitlines()
                if lines:
                    if lines[-1] != "\x1b[0m":
                        last_stdout_line = lines[-1]
                    elif len(lines) > 1:
                        last_stdout_line = lines[-2]
            process.wait()
        finally:
            process.stdout.close()
            process.stderr.close()
    else:
        import pty
        import select
        output_stream = sys.stdout.write
        master_fd, slave_fd = pty.openpty()
        process = subprocess.Popen(
                    command,
                    stdout=slave_fd, stderr=slave_fd,
                    shell=False, env=env, bufsize=1, text=True)

        try:
            while True:
                reads, _, _ = select.select([master_fd], [], [], 0.1)
                if master_fd in reads:
                    output = os.read(master_fd,
                                     1024).decode("utf-8",
                                                  errors="ignore")
                    if not output:
                        break
                    output_stream(output)
                    sys.stdout.flush()
                    lines = output.splitlines()
                    if lines:
                        if lines[-1] != "\x1b[0m":
                            last_stdout_line = lines[-1]
                        elif len(lines) > 1:
                            last_stdout_line = lines[-2]
                if process.poll() is not None:
                    break
        finally:
            os.close(master_fd)
            os.close(slave_fd)
            process.wait()
            last_stdout_line = _strip_ansi_escape_codes(
                last_stdout_line.strip())

    return last_stdout_line.endswith("Done!")

def cli_utility():
    args = sys.argv[1:]  # all arguments
    #print(f"args: {args}")

    env = os.environ.copy()
    env['launched_from_cli'] = 'True'
    return retrain_pipelines_local(
               command=" ".join(args),
               env=env
           )

