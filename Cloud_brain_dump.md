# Cloud brain dump:

Within this knowledge base, steps will be described by which everyone can set up communication with the Oracle, as well as reproduce the deploying process to the cloud. In general, I will highlight 3 global steps with the help of which everyone will be able to deploy, for example, a shard in the Oracle Cloud:

1. Configuration of the local environment for remote communication with Oracle Cloud
2. Creating a cluster on Oracle cloud
3. Cluster configuration for comfortable work (download images, connecting to local kubectl, rights, opening ports, etc.)

# 1. Configuration of the local environment for remote communication with Oracle cloud

Before we begin, I expect you to have the credentials to successfully log in to Oracle Cloud.

Communication with the Oracle is through oci-cli, so our task in this section is to set up the communication with the Oracle using the following steps.

**1.1 Install or upgrade oci-cli:**

`brew update && brew install oci-cli` or`brew update && brew upgrade oci-cli`

**1.2 Setup local config:**

`oci setup config`

This command will help generate the public and private keys and configure the `config` file. If you have already run this command before and have multiple profiles, but want to add a new profile, you can use the specified command again, but in the configuration process - do not OVERWRITE files, but simply update `config` with a new profile and create additional keys.

To configure the `config` file, you will need to log in to Oracle and take the next parameters: `user OCID`, and `tenancy OCID` also you should know your region.

After successful configuration and key generation, your config file should look like this:

```
[DEFAULT]
user=ocid1.user.oc1..aaaaaaaaljasfjgiuonldsqhmm3fstuhq6zyznm777tsvy5npfrdnv3ejqwe
fingerprint=04:40:f1:61:b3:77:e6:11:fb:11:11:d5:4c:fe:11:6a
key_file=/Users/nazaryanovets/.oci/oci_api_key.pem
tenancy=ocid1.tenancy.oc1..aaaaaaaaii0udpmtqhokgof6mqewbfipvbgu7ux4vvpm7gaghfnnuvgjqwe
region=us-sanjose-1
pass_phrase=example
```

Where key_file=path to your private key.

As a result, by default, you will be able to find all needed files in the `$HOME` directory under the `.oci` folder.

**1.3. Add public key to Oracle cloud:**

Copy your public key `cat $HOME/.oci/oci_api_key_public.pem | pbcopy` and then do the next cycle: `Log in into Oracle Cloud -> Profile -> My profile -> Resources -> Api Keys -> Add Api Keys -> Paste a public key -> Save it`

If all the previous steps are done correctly, you can run the following `oci os ns get`command from the terminal and as a result you should get something similar:

```
{
  "data": "axb4qezqa1q2"
}
```

This means that you can now communicate with the Oracle through `oci-cli`.

For more commands and `oci` capabilities, check out [Oracle docs](https://docs.oracle.com/en-us/iaas/tools/oci-cli/3.39.1/oci_cli_docs/)

# 2. Creating a cluster on Oracle

In general, you can create a cluster through `oci`, but I recommend doing it through Oracle UI because it is much easier for non-experienced users.

Before creating a cluster, you should create a new compartment. By default, you will already have a root compartment, but creating a cluster under the root compartment is a bad thing.
So, let's create `f1r3fly-dev` compartment under `root` with the next cycle: `Search resources -> Compartments -> Create Compartment -> Pick f1r3fly root compartment as a parent compartment -> create f1r3fly-dev compartment`

After it create new kubernetes cluster under `f1r3fly-dev` compartment with the next cycle

```
Oracle Cloud -> Log in -> Developers Services -> Kubernetes Clusters (OKE) -> Picked f1r3fly-dev compartment -> Create cluster -> Quick create with next params->:
Public Endpoint,
Managed Node type,
Public workers
VM.Standart.E3.Flex
Needed service limits,
Node count 1 
 -> Create Cluster
```

After these steps, it will take some time, as the cluster will start. You can view the launch process in the `Resources -> Work Request` tab.

# 3. Cluster configuration for comfortable work

In this section, I expect that you already have a Kubernetes cluster created on the Oracle cloud. In addition, you do not have to repeat all the steps described in this section, as they may be different depending on your needs.

3.1 **Local access to Kubernetes cluster**

Do the next cycle: `Developer services -> Kubernetes Clusters(OKE) -> Open created cluster -> Resources -> Quick Start -> Local access.`

After that, you should repeat the steps described on the `Local access` tab. The steps described in the specified tab will allow you to communicate with the cluster via `kubectl` and add new context under `kubectl config`.

As a result, you should be able to see new context inside your local `kubectl config`. To check it, you can use the next command `kubectl config view`. By default, you'll be looking at the context you just added with the `Local access` tab, so now you can use any kubectl commands and check deployments, and other things from your remote Oracle cluster.

3.2 **Load image inside Oracle registry**

Our helm files work by deploying the application based on the image, for our logic to work correctly, we need to put the image in any repository. At the moment we are using the native capabilities of Oracle and putting the image inside the `Container Registry`.

3.2.1 **Login inside docker**

Create an authentication token using the following cycle: `User -> My Profile -> Resources -> Auth Token -> Generate Token -> Save it`. Then you should go inside `Tenancy details` and get `Object storage namespace`.
After you have `Auth token` and `object storage namespace` use the next command:

`docker login <your_region>.ocir.io` and next credentials:
username: `object_storage_namespace/your_user's_email` ; password: `Auth token`

As a result you should see `Login Succeeded`

3.2.2 **Upload the image inside Oracle Registry**

After you have successfully logged in to docker, we need to select the image we want to upload to the Registry and create a `tag` for this image.

Use `docker images` command and take `image_id` for an image that you wanna upload. After that you should build the next command:

`docker tag <image_id> <region>.ocir.io.<object_storage_namespace><folder_name>/<image_name>:<tag>` and finnally push the image with the next command:

`docker push <region>.ocir.io.<object_storage_namespace><folder_name><image_name>:<tag>`

After that, as a result in Containers and Artifacts (Container Registry), you should see the uploaded image with your tag.

3.3 **Create dynamic group**

Create a dynamic group with OCID from your compartment. Do the next cycle: `Identity -> Domains -> Default Domain -> Dynamic Groups -> Create new one -> then create rule with f1r3fly-dev OCID`

As an example of a rule for the dynamic group: `ALL {resource.type = 'instance', instance.compartment.id = 'ocid1.compartment.oc1..aaaaaaaai76kz2wj4bwgjfkr6t4spqmpdpjnpo5fhgepuy2kkkf6lcvle7aa' }`

3.4 **Create Policies**

Well, we need policies so that our helm can read the image from the oracle and, as a result, perform the correct deployment based on the loaded image. So, do the next cycle: `Identity -> Policies -> Create Policy -> Add policy statement for needed dynamic group`

As an example of a policy statement: `Allow dynamic-group f1r3fly-dynamic-group to read repos in tenancy`

If the policies are configured correctly, we can deploy our helm files and communicate with our compartment: `oci compute instance list --compartment-id ocid1.compartment.oc1..aaaaaaaai76kz2wj4bwgjfkr6t4spqmpdpjnpo5fhgepuy2kkkf6lcvle7aa`

3.5 **Create Group for demployments**

This step is not mandatory, at least in the format in which we do it, but I want to show it because it insinuates the practice of creating a global user for all developers, through the prism of which all developers perform deployments.

Create group with next cycle: `Indenity -> Domains -> Default Domains -> Create Group for k8s deployments -> Assign users to group`

In case you have already configured your user, you do not need to perform additional steps, but if you want to create a user, for example, a deployment administrator for future deployments, you should not forget to configure it.

3.6 **Deployments**

In this section we will perform the final steps, I assume that you already have configured helm files with the necessary configurations (users, secrets, etc). I will show an example of the commands for deployment with 4 nodes (validators), but the logic for other clusters, for example with 8 or 16 nodes, will be similar.

1. Create namespace: `kubectl create ns  f1r3fly-4nodes`
2. Create a secret for your namespace: `kubectl create secret docker-registry <secret-name> --docker-server=<region>.ocir.io --docker-username=object_storage_namespace/your_user's_email --docker-password=<Auth token> --docker-email=<your_user's_email> -n f1r3fly-4nodes`

It should be said that secret name is just the name of our secret from the helm files, we can change it as we like. In addition, it is worth understanding that the secret is related to the namespace, that is, for each cluster that has its own namespace (f1r3fly-4nodes, f1r3fly-8nodes, etc), we must generate a new secret with the same command described above.

3. Install helm chart

3.7. **Create security list**

To open ports and communicate with our cluster (that is, to send requests), we need to configure a security list, for this next cycle: `Under VCN related to needed cluster -> Security Lists -> Create Security List -> Add Ingress and Egress rules to allow the TCP traffic for all ports`. If you have your unique ports, you may need to configure additional rules on unique ports instead of the entire range of ports.

3.8 **Create network security group**

To limit the incoming and outgoing traffic of our cluster, we need to create a network security group, for this do the next cycle: `Under VCN related to needed cluster -> Network Security Groups -> Create Security Group -> Add Ingress and Egress rules for incoming and outgoing traffic`

If the current step is not taken, the cluster will be public and vulnerable to potential dangers. It is worth saying that at the moment we are restricting communication with our clusters using a VPN and configuring appropriate rules.

### Summary

If you follow the steps I described, you will be able to deploy a Kubernetes cluster on Oracle without any problems. Also, I only described my way, you should understand that certain steps may differ in certain situations. In my case, I set up 3 clusters with 4, 8, and 16 nodes making the most of Oracle's default capabilities, such as cluster generation through the UI. However, this is not the only true way.

# Open Questions:

1. **Access restriction of incoming traffic.**
   Our cluster is public and we use Kubernetes with a reserved ports range. This means that the bad guys can send requests to our cluster. To fix it we should create Ingrees rules for each public ip's or create a VPN. A good approach would be to use a VPN. **[Fixed via VPN]**
2. **Random restart of the pods of the cluster.**
   It is worth doing research here, potentially these restarts may be related to bad requests from the Internet, after limiting traffic it seems that restarts do not occur. **[Fixed via additional pods resources]**
3. A potential question may be in the load. So far, we do not know how our cluster will behave under heavy load. However, this can potentially be solved with a Load Balancer. **[Potential issue]**

# TIPS:

**kubectl**:
When you add a new context (step 3.1) by default you start looking at it, however, you can switch between contexts with the command - `kubectl config use-context contextName`

**dashboard**:

For new users, it is convenient to use the dashboard, by default it is not deployed in the cluster, that is why if you want to add it, take the following steps:

1. `kubectl apply -f https://raw.githubusercontent.com/kubernetes/dashboard/v2.7.0/aio/deploy/recommended.yaml`
2. Run `kubectl proxy`
3. Open “[http://localhost:8001/api/v1/namespaces/kubernetes-dashboard/services/https:kubernetes-dashboard:/proxy/](http://localhost:8001/api/v1/namespaces/kubernetes-dashboard/services/https:kubernetes-dashboard:/proxy/)” in your browser.
4. `kubectl create sa kube-ds-viewer -n kubernetes-dashboard` (create service account)
5. `kubectl create clusterrolebinding kube-ds-viewer-role-binding --clusterrole=view --serviceaccount=kubernetes-dashboard:kube-ds-viewer`
6. `kubectl create token kube-ds-viewer -n kubernetes-dashboard` (copy token)
7. Use the copied token in your browser.

In addition, there is a great additional feature, which is metrics. With the help of metrics, we will be able to observe how many resources are used. To add metrics inside the dashboard, use the next command:

`kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml`

**stop/run kubernetes cluster**:

To stop a Kubernetes cluster managed by Oracle Kubernetes Engine (OKE) in Oracle Cloud Infrastructure (OCI), you will actually need to scale down the nodes in the node pool to zero. This effectively stops all compute instances in your cluster, stopping the cluster itself. Oracle doesn't offer a direct "stop" button for OKE clusters because Kubernetes clusters are designed to be highly available and stopping them isn't a typical operation. **Scaling down to zero nodes is an effective way to pause operations and reduce costs, especially when you do not need the cluster to be operational**. It's a suitable strategy for environments used intermittently, such as development or testing environments

If you wanna run a cluster, just make node pool >= 1 under cluster, and then do not forget to add **Network security groups**, without it, you can't connect to the cluster. After this operation node pool will have a new **public ip** for communication.

Helpfull commands:

check storage class:
`kubectl get sc -n f1r3fly-4nodes`

check persistent volume claim:
`kubectl get pvc -n f1r3fly-4nodes`

check persistent volumes:
`kubectl get pv -n f1r3fly-4nodes`

check all pod events:
`kubectl events f1r3fly-4nodes0-0 -n f1r3fly-4nodes`

describe pod:
`kubectl describe pod f1r3fly-4nodes0-0 -n f1r3fly-4nodes`

get all pods with system pods under namespace:
`kubectl get pods -n f1r3fly-4nodes -A`
