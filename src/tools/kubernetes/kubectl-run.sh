# 如果pod失败会返回失败码。这块可以如同docker run -it xx /bin/bash --  # 一样跑数据
set -e
kubectl run -i --tty --restart=Never --rm ${name} --generator=run-pod/v1 --labels ${labels} --image=${image} --attach -- /bin/sh test.sh