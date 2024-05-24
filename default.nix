{
  config,
  lib,
  autoAddDriverRunpath,
  cudaPackages,
  ...
}:
let
  inherit (cudaPackages)
    backendStdenv
    cuda_cudart
    cuda_nvcc
    cudaAtLeast
    cudaOlder
    cudatoolkit
    setupCudaHook
    ;
  inherit (lib) getDev getLib getOutput;
in
backendStdenv.mkDerivation {
  pname = "tdr-inverse";
  version = "0.0.1";

  src = ./.;

  strictDeps = true;

  nativeBuildInputs = [
    autoAddDriverRunpath
    cuda_nvcc
  ];

  buildInputs = [ cuda_cudart ];

  # buildPhase = ''
  #   make
  # '';

  installPhase = ''
    mkdir -p $out/bin
    install ./bin/inverse $out/bin
    install ./bin/cpu-inverse $out/bin
    install ./bin/tdr-inverse $out/bin
  '';

  meta = with lib; {
    name = "GPU-Based Matrix Inversion";
    description = ''
      Optimized GPU-Based Matrix Inversion
      Though The Use of Thread-Data Remapping
    '';
    homepage = "https://github.com/scrufulufugus/tdr-inverse";
    license = licenses.bsd3;
    platforms = platforms.all;
  };
}
