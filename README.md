# F1r3fly

### [What is F1r3fly?]([#what-is-this-project?])
- A decentralized, economic, censorship-resistant, public compute infrastructure and blockchain.
- Hosts and executes programs popularly referred to as "smart contracts".
- Trustworthy, scalable, concurrent, with proof-of-stake consensus and content delivery.
### How to get F1r3fly
- Access the F1r3fly Developer portal for tutorials, documentation, and project information.
- Run the public testnet to explore F1r3fly's capabilities.
### [Installation instructions](#Installation)
- Install F1r3fly using Docker, Debian/Ubuntu, RedHat/Fedora, or macOS.
- Run F1r3fly using Docker commands or as a standalone process.
### Running F1r3fly with multiple nodes
Connect multiple Docker RNodes to a user-defined network bridge for peer-to-peer communication and REPL capabilities.
### Using the REPL
- Invoke the REPL using the provided Docker command and interact with F1r3fly using Rholang language.
- Validate F1r3fly's functionality by executing a command in the REPL and observing the output on the rnode0 node.
### Peer node configuration
- Start a peer node using the specified Docker command and provide the bootstrap address of rnode0.
- Observe the successful communication between the main node and the peer node in the output of both nodes.
### Obtaining command options
- Use the --help option to get a comprehensive list of command-line options for RNode.
### Configuration file
- Specify RNode configuration parameters through a configuration file in HOCON format.
- View the example configuration file for a detailed explanation of each parameter.
### Development environment
- Compile the F1r3fly project using the provided command.
- Run the compiled binary using the specified command.
### Developer resources
- Refer to the developer guide for more in-depth instructions and documentation.
### Known issues
- Be aware of the known issues listed in the provided GitHub link.
### Reporting issues
- File any issues you encounter in the GitHub repository issue tracker.
### Acknowledgments
- Express gratitude to YourKit for their support of open-source projects.
### License information
- Generate a summary of licenses used by F1r3fly's dependencies using the provided command.


## What is this project?
[![Build Status](https://github.com/rchain/rchain/workflows/CI/badge.svg)](https://github.com/rchain/rchain/actions?query=workflow%3ACI+branch%3Astaging)
[![codecov](https://codecov.io/gh/rchain/rchain/branch/master/graph/badge.svg)](https://codecov.io/gh/rchain/rchain)

The open-source F1r3fly project is building a decentralized, economic,
censorship-resistant, public compute infrastructure and blockchain. It will
host and execute programs popularly referred to as “smart contracts”. It will
be trustworthy, scalable, concurrent, with proof-of-stake consensus and
content delivery.

[F1r3fly Developer](https://developer.rchain.coop/) features project-related
tutorials and documentation, project planning information, events calendar,
and information for how to engage with this project.

## Note on the use of this software
This code has not yet completed a security review. We strongly recommend that you do not use it in production or to transfer items of material value. We take no responsibility for any loss you may incur through the use of this code.

## Use the public testnet
The F1r3fly Cooperative maintains a public testnet running the latest version of RNode. Learn more at [F1r3fly public testnet information](https://rchain.atlassian.net/wiki/spaces/CORE/pages/678756429/RChain+public+testnet+information).

## How 
## Installation

### Docker

`$ docker pull rchain/rnode:latest`

### Debian/Ubuntu

1. Download a `.deb` package from the [releases page](https://github.com/rchain/rchain/releases/)
2. `$ sudo apt install ./rnode_<VERSION>.deb`, where `<VERSION>` is something like `0.9.18`

### RedHat/Fedora

1. Download a `.rpm` package from the [releases page](https://github.com/rchain/rchain/releases/)
2. `$ sudo rpm -U ./rnode_<VERSION>.noarch.rpm`, where `<VERSION>` is something like `0.9.18`

### macOS

1. Install Homebrew by following steps at the [Homebrew homepage](https://brew.sh/)
2. `$ brew install rchain/rchain/rnode`

## Running

Docker will be used in the examples port portability reasons, but running the
node as a standalone process is very similar.

To fetch the latest version of RNode from the remote Docker hub and run it (exit with `C-c`):

```sh
$ docker run -it -p 40400:40400 rchain/rnode:latest

# With binding of RNode data directory to the host directory $HOME/rnode 
$ docker run -v $HOME/rnode:/var/lib/rnode -it -p 40400:40400 rchain/rnode:latest
```

In order to use both the peer-to-peer network and REPL capabilities of the
node, you need to run more than one Docker RNode on the same host, the
containers need to be connected to one user-defined network bridge:

```bash
$ docker network create rnode-net

$ docker run -dit --name rnode0 --network rnode-net rchain/rnode:latest run -s

$ docker ps
CONTAINER ID   IMAGE                 COMMAND                  CREATED          STATUS          PORTS     NAMES
ef770b4d4139   rchain/rnode:latest   "bin/rnode --profile…"   23 seconds ago   Up 22 seconds             rnode0
```

To attach terminal to RNode logstream execute

```bash
$ docker logs -f rnode0
[...]
08:38:11.460 [main] INFO  logger - Listening for traffic on rnode://137200d47b8bb0fff54a753aabddf9ee2bfea089@172.18.0.2?protocol=40400&discovery=40404
[...]
```

A repl instance can be invoked in a separate terminal using the following command:

```bash
$ docker run -it --rm --name rnode-repl --network rnode-net rchain/rnode:latest --grpc-host rnode0 --grpc-port 40402 repl

  ╦═╗┌─┐┬ ┬┌─┐┬┌┐┌  ╔╗╔┌─┐┌┬┐┌─┐  ╦═╗╔═╗╔═╗╦  
  ╠╦╝│  ├─┤├─┤││││  ║║║│ │ ││├┤   ╠╦╝║╣ ╠═╝║  
  ╩╚═└─┘┴ ┴┴ ┴┴┘└┘  ╝╚╝└─┘─┴┘└─┘  ╩╚═╚═╝╩  ╩═╝
    
rholang $
```

Type `@42!("Hello!")` in REPL console. This command should result in (`rnode0` output):
```bash
Evaluating:
@{42}!("Hello!")
```

A peer node can be started with the following command (note that `--bootstrap` takes the listening address of `rnode0`):

```bash
$ docker run -it --rm --name rnode1 --network rnode-net rchain/rnode:latest run --bootstrap 'rnode://8c775b2143b731a225f039838998ef0fac34ba25@rnode0?protocol=40400&discovery=40404' --allow-private-addresses --host rnode1
[...]
15:41:41.818 [INFO ] [node-runner-39      ] [coop.rchain.node.NodeRuntime ] - Starting node that will bootstrap from rnode://8c775b2143b731a225f039838998ef0fac34ba25@rnode0?protocol=40400&discovery=40404
15:57:37.021 [INFO ] [node-runner-32      ] [coop.rchain.comm.rp.Connect$ ] - Peers: 1
15:57:46.495 [INFO ] [node-runner-32      ] [c.r.c.util.comm.CommUtil$    ] - Successfully sent ApprovedBlockRequest to rnode://8c775b2143b731a225f039838998ef0fac34ba25@rnode0?protocol=40400&discovery=40404
15:57:50.463 [INFO ] [node-runner-40      ] [c.r.c.engine.Initializing    ] - Rholang state received and saved to store.
15:57:50.482 [INFO ] [node-runner-34      ] [c.r.casper.engine.Engine$    ] - Making a transition to Running state.
```

The above command should result in (`rnode0` output):
```bash
15:57:37.021 [INFO ] [node-runner-42      ] [c.r.comm.rp.HandleMessages$  ] - Responded to protocol handshake request from rnode://e80faf589973c2c1b9b8441790d34a9a0ffdd3ce@rnode1?protocol=40400&discovery=40404
15:57:37.023 [INFO ] [node-runner-42      ] [coop.rchain.comm.rp.Connect$ ] - Peers: 1
15:57:46.530 [INFO ] [node-runner-43      ] [c.r.casper.engine.Running$   ] - ApprovedBlock sent to rnode://e80faf589973c2c1b9b8441790d34a9a0ffdd3ce@rnode1?protocol=40400&discovery=40404
15:57:48.283 [INFO ] [node-runner-43      ] [c.r.casper.engine.Running$   ] - Store items sent to rnode://e80faf589973c2c1b9b8441790d34a9a0ffdd3ce@rnode1?protocol=40400&discovery=40404
```

To get a full list of options rnode accepts, use the `--help` option:

```sh
$ docker run -it --rm rchain/rnode:latest --help
```

### Configuration file

Most of the command line options can be specified in a configuration file.

The default location of the configuration file is the data directory. An
alternative location can be specified with the command line option `--config-file <path>`.

The format of the configuration file is [HOCON](https://github.com/lightbend/config/blob/master/HOCON.md).

The [defaults.conf](node/src/main/resources/defaults.conf) configuration file shows all options and default values.

Example configuration file:

```yml
standalone = false

protocol-server {
  network-id = "testnet"
  port = 40400
}

protocol-client {
  network-id = "testnet"
  bootstrap = "rnode://de6eed5d00cf080fc587eeb412cb31a75fd10358@52.119.8.109?protocol=40400&discovery=40404"
}

peers-discovery {
  port = 40404
}

api-server {
  host = "my-rnode.domain.com"
  port-grpc-external = 40401
  port-grpc-internal = 40402
  port-http = 40403
  port-admin-http = 40405
}

storage {
  data-dir = "/my-data-dir"
}

casper {
  fault-tolerance-threshold = 1
  shard-name = root
  finalization-rate = 1
}

metrics {
  prometheus = false
  influxdb = false
  influxdb-udp = false
  zipkin = false
  sigar = false
}

dev-mode = false
```

## Development

Compile the project with:

```bash
$ sbt clean compile

# With executable and Docker image
$ sbt clean compile stage docker:publishLocal
```

Run the resulting binary with:

```bash
$ ./node/target/universal/stage/bin/rnode
```

For more detailed instructions, see the [developer guide](DEVELOPER.md).

## Cloud
We use Oracle! We have several environments for different purposes. For more detailed instructions, see the [Cloud_brain_dump](Cloud_brain_dump.md).

## Caveats and filing issues

### Caveats

During this pre-release phase of the F1r3fly software, there are some [known issues](https://github.com/rchain/rchain/issues?q=is%3Aopen+is%3Aissue+label%3Abug).

### Filing Issues

File issues in GitHub repository issue tracker: [File a bug](https://github.com/rchain/rchain/issues/new/choose).

## Acknowledgements

We use YourKit to profile rchain performance.  YourKit supports open source
projects with its full-featured Java Profiler.  YourKit, LLC is the creator of
<a href="https://www.yourkit.com/java/profiler/">YourKit Java Profiler</a> and
<a href="https://www.yourkit.com/.net/profiler/">YourKit .NET Profiler</a>,
innovative and intelligent tools for profiling Java and .NET applications.

## Licence information

To get summary of licenses being used by the F1r3fly's dependencies, simply run
`sbt node/dumpLicenseReport`. The report will be available under
`node/target/license-reports/rnode-licenses.html`
