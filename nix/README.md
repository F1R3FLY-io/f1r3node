# RChain project setup/build using *nix*

This document describes developer setup and build commands for RChain project using _nix-shell_.

**Note:** _default.nix_ file in this directory is not part of this document. Using _shell.nix_

## Nix

Install Nix https://nixos.org/download.html

### MacOS
```
sh <(curl -L https://nixos.org/nix/install)
```

The `nix` installation may run some commands as root and you may need to run the following to set permissions correctly:

```
chown -R $(id -nu):$(id -ng) ~/.local
```

In order to allow the `direnv` to use the project environment, you may need to run the following in the project root:

```
direnv allow
```

### Windows

1. Install Windows Subsystem for Linux -> https://learn.microsoft.com/en-us/windows/wsl/install#install-wsl-command
2. Install Ubuntu -> `wsl --install Ubuntu`
3. Open Ubuntu terminal app or restart Windows terminal app and add ubuntu profile
4. In ubuntu terminal, install nix -> https://nixos.org/download.html#nix-install-linux
```
sh <(curl -L https://nixos.org/nix/install) --daemon
```


Test `nix` installation

```sh
# Install example Hello World program
nix-env -i hello

# Execute hello program
hello
# Hello, world!

# Unistall Hello World program
nix-env -e hello

# Run Hello World nix-shell
nix-shell -p hello

[nix-shell:~]$ hello
Hello, world!

[nix-shell:~]$ exit

$ hello
hello: command not found
```

More info on Nix -> https://nixos.org/manual/nix/stable/introduction.html

## CI
We have a CI server setup with NixOs using shell script to pull the release every hour. Check with Discord to find the current location.

## RNode build (in nix-shell)

Enter nix-shell using zsh and setup keys: 
```
nix-shell --pure nix/shell.nix --run zsh
# enter ssh-agent password or permanently add keys; this may have to be done each time since nix-shell may be ephemeral
ssh-add -K ~/.ssh/<key-name>
```

```sh
# Compile
[nix-shell:~/src/rchain]$ sbt compile

# Compile with tests
[nix-shell:~/src/rchain]$ sbt test:compile

# Compile and create local executable
# path: rchain/node/target/universal/stage/bin/rnode
[nix-shell:~/src/rchain]$ sbt stage

# Clean project (except bnfc generated Java code)
[nix-shell:~/src/rchain]$ sbt clean

[nix-shell:~/src/rchain]$ sbt clearCaches

# Clean, compile, and create local executable
[nix-shell:~/src/rchain]$ sbt clean compile stage

# List rnode options
[nix-shell:~/src/rchain]$ rnode

# Run stand-alone node
[nix-shell:~/src/rchain]$ rnode run --standalone

# Check node status
[nix-shell:~/src/rchain]$ rnode status

# Run .rho files (have running node in separate shell)
[nix-shell:~/src/rchain]$ rnode eval <path_to_file> # (i.e. rholang/examples/hello_world_again.rho)

# Run .rho files (have ip address and port of node)
[nix-shell:~/src/rchain]$ rnode --grpc-host <ip_address> --grpc-port <internal_port> eval <path_to_file> # (i.e. rholang/examples/hello_world_again.rho)
```

Can also use JS implementation for eval -> https://github.com/spreston8/rnode-repl


### `sbt`  interactive mode

```sh
# Enter sbt interactive mode
sbt

# Compile
sbt:rchain> compile

# Compile with tests
sbt:rchain> test:compile

# Compile and create local executable
# path: ./node/target/universal/stage/bin/rnode
sbt:rchain> stage

# Clean project (except bnfc generated Java code)
sbt:rchain> clean

sbt:rchain> clearCaches
```



### Reset Git repository to a clean state

**WARNING: this will remove all non-versioned files from your local repository folder**

```sh
git clean -fdx
```

## Additional Notes

Nix package versions: https://lazamar.co.uk/nix-versions/

### Helpful nix commands

- Lists basic nix information: `nix-shell -p nix-info --run "nix-info -m"`
- List specific package version: `readlink -f $( which <package_name> )`
- Delete unused packages: `nix-collect-garbage`

### VS Code within nix-shell

- Add `vscode` to package list in `nix/shell.nix` and also add `{ allowUnfree = true; }` in your `~/.config/nixpkgs/config.nix`

### Information for WSL and SSH

- Check Git is installed: `git --version` if not then: `sudo apt-get install git`
- Configure .gitconfig
- Copy ssh keys to Ubuntu: `cp -r /mnt/c/Users/<username>/.ssh ~/.ssh` or `cp -r /mnt/c/Users/<username>/.ssh/<file-name> ~/.ssh`
- - If bad permission, then change for specefied file: `chmod 600 ~/.ssh/<file-name>`
- Test connection: `ssh -T git@github.com`
- Create src directory: `mkdir src && cd src` and clone project `git clone git@github.com:rchain/rchain.git`

## Using with NIX-ENV (old docs)

## Java 11

```sh
sudo update-alternatives --config java
# If necessary install Java 11 version
sudo apt install default-jdk
```

## Nix

Install Nix https://nixos.org/download.html

```sh
curl -L https://nixos.org/nix/install | sh
```

Test `nix` installation

```sh
# Install example Hello World program
nix-env -i hello
# Execute hello program
hello
# Hello, world!
# Unistall Hello World program
nix-env -e hello
```

## sbt

Install Scala build tool `sbt`

```sh
sudo apt install sbt
```

## BNFC

Install `jflex` and `bnfc` with *nix*

```sh
# Install BNFC and jflex with nix
# - jflex v1.7.0 with ghc 8.6.5
nix-env -i jflex -iA haskellPackages.BNFC --file https://github.com/NixOS/nixpkgs-channels/archive/nixos-20.03.tar.gz
# Uninstall
nix-env -e jflex BNFC
# Install in case of error (Ubuntu)
sudo apt-get install libgmp3-dev
```

## RNode build

```sh
# Compile
sbt compile
# Compile with tests
sbt test:compile
# Compile and create local executable
# path: rchain/node/target/universal/stage/bin/rnode
sbt stage
# Compile Docker image
sbt docker:publishLocal
# Clean project (except bnfc generated Java code)
sbt clean
```

Default memory limits may not be sufficient so additional options for _sbt_ can be specified. They can be added to `.bashrc` file.

Increase heap memory and thread stack size. Disable _supershell_ if empty lines are printed in _sbt_ output.

```sh
export SBT_OPTS="-Xmx4g -Xss2m -Dsbt.supershell=false"
```


### `sbt`  interactive mode

```sh
# Enter sbt interactive mode
sbt
# sbt entering interactive mode
# sbt:rchain>
# Compile
compile
# Compile with tests
test:compile
# Compile and create local executable
# path: ./node/target/universal/stage/bin/rnode
stage
# Compile Docker image
docker:publishLocal
# Clean project (except bnfc generated Java code)
clean
```

### Reset Git repository to a clean state

**WARNING: this will remove all non-versioned files from your local repository folder**

```sh
git clean -fdx
```
