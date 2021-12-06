import datetime
import time

import paramiko


class SSH:
    def __init__(self, hostname, username, password, port=22):
        self.client = paramiko.SSHClient()
        self.client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        self.client.connect(hostname=hostname, port=port, username=username, password=password, timeout=20)
        self.channel = self.client.invoke_shell()
        self.buff = ''
        # return str(self.channel.recv(1024), 'utf-8')

    def exec_command(self, cmd):
        return self.client.exec_command(cmd)

    def get_command_results(self):
        ## http://joelinoff.com/blog/?p=905
        interval = 0.1
        maxseconds = 10
        maxcount = maxseconds / interval
        bufsize = 1024

        # Poll until completion or timeout
        # Note that we cannot directly use the stdout file descriptor
        # because it stalls at 64K bytes (65536).
        input_idx = 0
        timeout_flag = False
        start = datetime.datetime.now()
        start_secs = time.mktime(start.timetuple())
        output = ''
        self.channel.setblocking(1)
        while True:
            if self.channel.recv_ready():
                data = self.channel.recv(bufsize).decode('ascii')
                print(data)
                output += data

            if self.channel.exit_status_ready():
                break

            # Timeout check
            now = datetime.datetime.now()
            now_secs = time.mktime(now.timetuple())
            et_secs = now_secs - start_secs
            if et_secs > maxseconds:
                timeout_flag = True
                break

            rbuffer = output.rstrip(' ')
            if len(rbuffer) > 0 and (rbuffer[-1] == '#' or rbuffer[-1] == '>'):  ## got a Cisco command prompt
                break
            # 为网络延迟给的时间
            time.sleep(0.200)
        if self.channel.recv_ready():
            data = self.channel.recv(bufsize)
            output += data.decode('ascii')
        return output

    def send_command_and_get_response(self, cmd, password=None):
        self.channel.setblocking(1)
        if not cmd.endswith("\n"):
            cmd += "\n"
        self.channel.sendall(cmd)
        results = self.get_command_results()
        if not results.find('y/n') == -1:
            self.channel.sendall(b'y\n')
            results = self.get_command_results()
        if not results.find('yes/no') == -1:
            self.channel.sendall(b'yes\n')
            results = self.get_command_results()
        if results.endswith('\'s password: '):
            self.channel.sendall(password + '\n')
            results = self.get_command_results()
        return results

    def close(self):
        self.client.close()
