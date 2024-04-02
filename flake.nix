{
  description = "tdr-inverse Research";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils, ... }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; config.allowUnfree = true; };
      in
      {
        # TODO: Have to override with no CC since we're building with host CUDA
        devShells.default = pkgs.mkShell.override { stdenv = pkgs.stdenvNoCC; } {
          packages = with pkgs; [
            clang-tools # LSP code server
            #cudaPackages_12_3
          ];
        };
      });
}
