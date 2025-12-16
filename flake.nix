{
  inputs = {
    typelevel-nix.url = "github:typelevel/typelevel-nix";
    nixpkgs.follows = "typelevel-nix/nixpkgs";
    oldNixpkgs.url = "github:NixOS/nixpkgs/nixos-23.05";
    flake-utils.follows = "typelevel-nix/flake-utils";
    rust-overlay.url = "github:oxalica/rust-overlay";
    nixpkgs-23.url = "github:NixOS/nixpkgs/d4f247e89f6e10120f911e2e2d2254a050d0f732"; #https://www.nixhub.io/packages/tree-sitter
    naersk.url = "github:nix-community/naersk";
    cross-src.url = "github:cross-rs/cross";
    cross-src.flake = false;
  };

  outputs = { self, nixpkgs, oldNixpkgs, flake-utils, typelevel-nix, rust-overlay, nixpkgs-23, naersk, cross-src }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        graalOverlay = final: prev: rec {
          holyGraal = final.jdk17;
          jdk = holyGraal;
          jre = holyGraal;
        };
        ammOverlay = final: prev: {
          hematite = prev.ammonite.overrideAttrs rec {
            version = "2.5.11";
            src = builtins.fetchurl {
              url =
                "https://github.com/lihaoyi/Ammonite/releases/download/${version}/2.12-${version}";
              sha256 = "0ycwdcpprfd195i5f2ds03z2vpifv8fky6i9wh9v328z0glnjwrg";
            };
          };
        };
        overlays = [ typelevel-nix.overlay (import rust-overlay) graalOverlay ammOverlay ];
        pkgs = import nixpkgs {
          inherit system overlays;
          config.allowUnfree = true;
        };
        pkgs23 = import nixpkgs-23 { inherit system; };
        
        naersk-lib = naersk.lib."${system}".override {
          cargo = pkgs.rust-bin.stable.latest.minimal;
          rustc = pkgs.rust-bin.stable.latest.minimal;
        };
        
        cross = naersk-lib.buildPackage {
          pname = "cross";
          root = cross-src;
          nativeBuildInputs = with pkgs; [ pkg-config ];
          buildInputs = with pkgs; [ openssl ];
        };
      in
      with pkgs;
      {
        devShells.default = devshell.mkShell {
          commands = [
            # Note: Using system Docker client (Docker Desktop) instead of nix package
            # to ensure API version compatibility with the Docker daemon
            {
              name = "protoc";
              package = protobuf;
              help = "Protocol Buffers compiler";
            }
            {
              name = "rustup";
              package = rustup;
              help = "The Rust toolchain installer";
            }
          ] ++ (if system != "x86_64-darwin" && system != "aarch64-darwin" then [{
              name = "gcc";
              package = gcc;
              help = "GNU C compiler";
            }] else []) ++ [
            {
              name = "make";
              package = gnumake;
              help = "GNU Make build tool";
            }
            {
              name = "pkg-config";
              package = pkg-config;
              help = "Helper tool for compiling applications";
            }
            {
              name = "bnfc";
              package = "haskellPackages.BNFC";
              help = "EBNF parser generator targeting several languages";
            }
            {
              name = "jflex";
              package = "jflex";
              help = "Java lexical analyzer generator";
            }
            {
              name = "amm";
              package = hematite;
              help = "Ammonite REPL for Scala";
            }
            {
              name = "grpcurl";
              package = grpcurl;
              help = "CURL-inspired CLI for gRPC services";
            }
            {
              name = "minikube";
              package = minikube;
              help = "Local Kubernetes cluster CLI";
            }
            {
              name = "kubectl";
              package = kubectl;
              help = "Kubernetes CLI";
            }
            {
              name = "oci";
              package = oci-cli;
              help = "Oracle Cloud CLI";
            }
            {
              name = "dhall";
              package = dhall;
              help = "Dhall configuration language";
            }
            {
              name = "dhall-to-yaml-ng";
              package = dhall-yaml;
              help = "Dhall-to-YAML utility";
            }
            {
              name = "tree-sitter";
              package = pkgs23.tree-sitter;
              help = "Parser generator tool and incremental parsing library";
            }
            {
              name = "cross";
              package = cross;
              help = "Cross-compilation of Rust projects made easy";
            }
            {
              name = "python3";
              package = python313;
              help = "Python 3.13 interpreter for integration tests";
            }
            {
              name = "pipenv";
              package = pipenv;
              help = "Python virtual environment manager";
            }
            {
              name = "pyenv";
              package = pyenv;
              help = "Python version manager";
            }
          ];
          imports = [ typelevel-nix.typelevelShell ];
          name = "f1r3fly-shell";
          typelevelShell = {
		        jdk.package = holyGraal;
          };
          env = [
              {
                name = "PROTOBUF_LOCATION";
                value = "${pkgs.protobuf}";
              }
              {
                name = "PROTOC";
                value = "${pkgs.protobuf}/bin/protoc";
              }
              {
                name = "PROTOC_INCLUDE";
                value = "${pkgs.protobuf}/include";
              }
              {
                name = "OPENSSL_STATIC";
                value = "1";
              }
              {
                name = "OPENSSL_INCLUDE_DIR";
                value = "${pkgs.openssl.dev}/include";
              }
              {
                name = "OPENSSL_LIB_DIR";
                value = "${pkgs.openssl.out}/lib";
              }
              {
                name = "BLOOP_JAVA_OPTS";
                value = "-Xmx4G -XX:+UseZGC -Xss4m -Xms1g";
              }
            ];
        };
      }
    );
}