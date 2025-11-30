{
  inputs = {
    typelevel-nix.url = "github:typelevel/typelevel-nix";
    nixpkgs.follows = "typelevel-nix/nixpkgs";
    oldNixpkgs.url = "github:NixOS/nixpkgs/nixos-23.05";
    # Newer nixpkgs for Docker 28.x compatibility
    newNixpkgs.url = "github:NixOS/nixpkgs/nixos-24.11";
    flake-utils.follows = "typelevel-nix/flake-utils";
    rust-overlay.url = "github:oxalica/rust-overlay";
  };

  outputs = { self, nixpkgs, oldNixpkgs, newNixpkgs, flake-utils, typelevel-nix, rust-overlay }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        # Import newer nixpkgs for Docker
        newPkgs = import newNixpkgs { inherit system; };
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
      in
      with pkgs;
      {
        devShells.default = devshell.mkShell {
          commands = [
            {
              name = "docker";
              package = newPkgs.docker;
            }
            {
              name = "cargo";
              package = "rust-bin.stable.latest.default";
              help = "The Rust package management tool";
            }
            {
              name = "rustc";
              package = "rust-bin.stable.latest.default";
              help = "The Rust compiler";
            }
            { name = "rust-analyzer";
              package = "rust-analyzer-unwrapped";
              help = "Language server for Rust";
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
              name = "helm";
              package = kubernetes-helm;
              help = "Helm is a tool that streamlines installing and managing Kubernetes applications";
            }
          ];
          imports = [ typelevel-nix.typelevelShell ];
          name = "f1r3fly-shell";
          typelevelShell = {
		        jdk.package = holyGraal;
          };
        };
      }
    );
}
