设置集群pods拉取docker权限

设置相应的secret拉取相应的docker，默认的仓库账号对某些命名空间无权限

```
kubectl create secret docker-registry registry-secret --namespace=default --docker-server=http://xxx.cn --docker-username="" --docker-password=""
```

```yaml
imagePullSecrets:
  - name: registry-secret
apiVersion: apps/v1
kind: DaemonSet
metadata:
  labels:
    name: hdfscli-env
  name: hdfscli-env
spec:
  selector:
    matchLabels:
      name: hdfscli-env
  template:
    metadata:
      labels:
        name: hdfscli-env
    spec:
      imagePullSecrets:
        - name: registry-secret
      containers:
        - image: xxx
          imagePullPolicy: IfNotPresent
          name: "hdfscli-env"
          stdin: true
          tty: true
          securityContext:
            privileged: true
          volumeMounts:
            - mountPath: /data
              name: data
      hostIPC: false
      hostNetwork: true
      hostPID: false
      nodeSelector:
        beta.kubernetes.io/arch: arm64
      volumes:
        - hostPath:
            path: /data
          name: data
```

gpu机器控制集群

```
docker run -itd --rm -uroot --name sv_test -v xxx:/.kube/config  --entrypoint='' xxx bash
```

```
docker run -itd --rm -uroot --name sv_test -v xxx:/.kube/config   --entrypoint='' xxx bash
vim ~/.bash_profile
export KUBECONFIG=$HOME/.kube/config-work-dev:$HOME/.kube/config-work-prod
source ~/.bash_profile
echo $KUBECONFIG
kubectl config get-contexts
kubectl config current-context
kubectl config use-context kubernetes-dev
```