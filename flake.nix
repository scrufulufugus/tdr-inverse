{
  description = "tdr-inverse Research";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils, ... }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config = {
            allowUnfree = true;
            cudaSupport = true;
            #cudaCapabilities = [ "7.5" "8.6"  ];
            cudaForwardCompat = false;
          };
        };
      in
      {
        packages.default = pkgs.callPackage ./. {
          stdenv = pkgs.gcc11Stdenv;
          cudaPackages = pkgs.cudaPackages_12_3;
        };

        # TODO: Have to override with no CC since we're building with host CUDA
        #devShells.default = pkgs.mkShell.override { stdenv = pkgs.stdenvNoCC; } {
        devShells.default = pkgs.mkShell.override { stdenv = pkgs.gcc11Stdenv; } {
          packages = with pkgs; [
            clang-tools # LSP code server
            cudaPackages_12_3.cudatoolkit
            cudaPackages_12_3.cuda_gdb
          ];
        };
      });
}
