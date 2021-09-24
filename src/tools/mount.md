```text
-*- coding: UTF-8 -*-
Author: https://github.com/Guanyan1996
         ┌─┐       ┌─┐
      ┌──┘ ┴───────┘ ┴──┐
      │                 │
      │       ───       │
      │  ─┬┘       └┬─  │
      │                 │
      │       ─┴─       │
      │                 │
      └───┐         ┌───┘
          │         │
          │         │
          │         │
          │         └──────────────┐
          │                        │
          │                        ├─┐
          │                        ┌─┘
          │                        │
          └─┐  ┐  ┌───────┬──┐  ┌──┘
            │ ─┤ ─┤       │ ─┤ ─┤
            └──┴──┘       └──┴──┘
                神兽保佑
                代码无BUG!
```

```shell
# nfs-server:
# install nfs on dev machine:
# https://vitux.com/install-nfs-server-and-client-on-ubuntu/
sudo apt install nfs-kernel-server
sudo chown nobody:nogroup /home/rjhuang/body_on_ipc
sudo exportfs -a
sudo systemctl restart nfs-kernel-server
echo '${mount_local_path} *(rw,sync,no_subtree_check,no_root_squash)' /etc/exports

# nfs-client:
mkdir -p ${local_mount_path}
mount -t nfs -o nolock -o tcp -o rsize=32768,wsize=32768 ${nfs-server-ip} ${local_mount_path}
```