### 如何使用docker

> 目的: agent docker存在局限性，使用的时候，只能在相应的master节；例如，我需要在指定slave节点使用docker,通过agent无法做到。
```groovy
withRegistry("${regisrty_url}","user-passwd设置对应的ID"){
    docker.image("${docker_image}").inside("-v -d -e等"){
    ${docker cmd}
    }
}
```
