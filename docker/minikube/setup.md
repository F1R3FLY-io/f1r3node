# Minikube installation

## Init local minikube
Create Minikube (Kubernetes single node) inside Docker container and expose ports:
```sh
minikube start \
    --extra-config=apiserver.service-node-port-range=30000-50000 \
    --ports=30000,30001,30002,30003,30004,30005,30010,30011,30012,30013,30014,30015,30016,30017,30018,30019,30020,30021,30022,30023,30024,30025,30026,30027,30028,30029,30030,30031,30032,30033,30034,30035,30036,30037,30038,30039,30040,30041,30042,30043,30044,30045 \
    --driver=docker \
    --addons='dashboard,metrics-server' \
    --cpus=8 \
    --memory=6g
```

## Pull `rnode` into Minikube
**This step needed if `f1r3flyindustries/f1r3fly-rust-node:latest` preset as local Docker image only. If the docker image has been published into remote Docker registry, skip this section.**

Load `f1r3flyindustries/f1r3fly-rust-node:latest` Docker image inside Minikube cache
```sh
minikube image load f1r3flyindustries/f1r3fly-rust-node:latest
```
Check the image list. `ghcr.io/f1r3fly-io/rnode` should be listed in the table
```sh
minikube image list --format=table
```
If `load` command failed (it's possible, [here is an open issue at GitHub](https://github.com/kubernetes/minikube/issues/18021)), use alternative mathod via file: store image into the file and load it from the file
```sh
docker image save f1r3flyindustries/f1r3fly-rust-node:latest -o rnode.tar && \
    minikube image load rnode.tar && \
    rm rnode.tar
```
Check the image list again using the command above.
## Check all system pods
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

## Deploy RNode

Deploy RNode into Minikube:
```sh
cd docker/minikube || true
helm upgrade --install f1r3fly ../helm/f1r3fly -n f1r3fly --create-namespace -f ./minikube-values.yaml
```

### Check status
Expose ports
```sh
kubectl port-forward service/f1r3fly0 40400:40400 40401:40401 40402:40402 40403:40403 40404:40404 40405:40405 -n f1r3fly
```
Expose ports for observer node
```sh
kubectl port-forward service/f1r3fly1 40410:40400 40411:40401 40412:40402 40413:40403 40414:40404 40415:40405 -n f1r3fly
```

Hit the endpoint into differnt terminal session
```
curl -s localhost:40403/api/status | jq .
```
The expected result should be like
```json
{
  "version": {
    "api": "1",
    "node": "RChain Node 1.0.0-SNAPSHOT (29aa391c6948c2c1df533b4cd35794d65eab04b5)"
  },
  "address": "rnode://c93bc07f3d141459356791f9eaaa05f4cec49c0b@f1r3fly0-0.f1r3fly0?protocol=40400&discovery=40404",
  "networkId": "f1r3fly",
  "shardId": "root",
  "peers": 1,
  "nodes": 3,
  "minPhloPrice": 1
}
```

## Open Dashboard
Open a dashboard, then picked f1r3fly namespace and check memory, cpu, and other information.
```sh
minikube dashboard
```

## Restart pods
Command to restart all pods if needed
```sh
kubectl delete pods f1r3fly0-0 f1r3fly1-0
```

## Logs
Bootstrap logs
```sh
kubectl logs f1r3fly0-0 | tail -n 50
```
Observer logs
```sh
kubectl logs f1r3fly1-0 | tail -n 50
```

## Undeploy RNode
Undeploy RNode from Minikube:
```sh
helm uninstall f1r3fly -n f1r3fly

```

## Cleanup
Delete local Minikube if needed:
```sh
minikube delete
```
