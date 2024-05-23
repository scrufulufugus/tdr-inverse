{
  config,
  stdenv,
  lib,
  cudaPackages ? { },
  ...
}:

stdenv.mkDerivation {
  pname = "tdr-inverse";
  version = "0.0.1";

  src = ./.;

  nativeBuildInputs = [
    cudaPackages.cudatoolkit
  ];

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
