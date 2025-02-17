# Minikube installation

## Init local minikube
Create Minikube with 4 validators(Kubernetes single node) inside Docker container and expose ports:
```sh
minikube start \
    --extra-config=apiserver.service-node-port-range=40000-50000 \
    --ports=40000,40001,40002,40003,40004,40010,40011,40012,40013,40014,40400,40401,40402,40403,40404,40410,40411,40412,40413,40414,40420,40421,40422,40423,40424,40430,40431,40432,40433,40434,40440,40441,40442,40443,40444 \
    --driver=docker \
    --addons='dashboard,metrics-server' \
    --cpus=8 \
    --memory=6g
```
Create Minikube with 8 validators(Kubernetes single node) inside Docker container and expose ports:
```sh
minikube start \
    --extra-config=apiserver.service-node-port-range=40000-50000 \
    --ports=40000,40001,40002,40003,40004,40010,40011,40012,40013,40014,40400,40401,40402,40403,40404,40410,40411,40412,40413,40414,40420,40421,40422,40423,40424,40430,40431,40432,40433,40434,40440,40441,40442,40443,40444 \
    --driver=docker \
    --addons='dashboard,metrics-server' \
    --cpus=10 \
    --memory=10g
```
## Pull `rnode` into Minikube
**This step needed if `ghcr.io/f1r3fly-io/rnode:latest` preset as local Docker image only. If the docker image has been published into remote Docker registry, skip this section.**

Load `ghcr.io/f1r3fly-io/rnode:latest` Docker image inside Minikube cache
```sh
minikube image load ghcr.io/f1r3fly-io/rnode:latest
```
Check the image list. `ghcr.io/f1r3fly-io/rnode` should be listed in the table
```sh
minikube image list --format=table
```
If `load` command failed (it's possible, [here is an open issue at GitHub](https://github.com/kubernetes/minikube/issues/18021)), use alternative mathod via file: store image into the file and load it from the file
```sh
docker image save ghcr.io/f1r3fly-io/rnode:latest -o rnode.tar && \
    minikube image load rnode.tar && \
    rm rnode.tar
```
Check the image list again using the command above.
## Simple test
Get a list of running system pods and check statuses:
```sh
kubectl get pods -A
```
If the output looks like the example below, Minikube got started successfully:
```
NAMESPACE     NAME                               READY   STATUS    RESTARTS      AGE
kube-system   coredns-5dd5756b68-pt8sz           1/1     Running   0             30m
kube-system   etcd-minikube                      1/1     Running   0             30m
kube-system   kube-apiserver-minikube            1/1     Running   0             30m
kube-system   kube-controller-manager-minikube   1/1     Running   0             30m
kube-system   kube-proxy-8p45l                   1/1     Running   0             30m
kube-system   kube-scheduler-minikube            1/1     Running   0             30m
kube-system   storage-provisioner                1/1     Running   1 (30m ago)   30m
```

## Open Dashboard
Open a dashboard, then picked f1r3fly namespace and check memory, cpu, and other information.
```sh
minikube dashboard
```
## Cleanup
Delete local Minikube if needed:
```sh
minikube delete
```
