```
目的：可在任意机器调用对应的k8s集群进行操作。
涉及到插件：https://plugins.jenkins.io/kubernetes-cli/
涉及docker: bitnami/kubectl:latest
```

> credentialsId为secrets file类型，上传/root/.kube/config文件进去之后即可。

> kubectl这个docker,在使用时，不加--entrypoint=''会error, not run except docker cmd.

```
pipeline {
  agent{ label '' }
  stages{
      stage('Apply Kubernetes files') {
        steps{
            script{
                docker.withRegistry('https://registry.xxx.cn/', "user") {
                   docker.image("registry.xxx.cn/kubernetes/kubectl:latest").inside("--entrypoint=''") {
                        withKubeConfig([credentialsId: 'IDC-3559-k8s-config']) {
                          sh 'kubectl get pods'
                        }
                    }
                }
            }
        }
      }
  }
}
```