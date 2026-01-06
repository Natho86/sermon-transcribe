{
  description = "sermon-transcribe dev shell";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
  };

  outputs = { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs {
        inherit system;
        config.allowUnfree = true;
      };

      cudaLibs = with pkgs.cudaPackages; [
        cuda_cudart
        cudnn
        libcublas
      ];

      baseLibs = [
        pkgs.stdenv.cc.cc.lib
        pkgs.zlib
        pkgs.ffmpeg
      ];
    in {
      devShells.${system}.default = pkgs.mkShell {
        packages = [
          (pkgs.python311.withPackages (ps: with ps; [
            pip
            setuptools
            wheel
          ]))
          pkgs.ffmpeg
          pkgs.zlib
        ] ++ cudaLibs;

        shellHook = ''
          export CUDA_HOME=${pkgs.cudaPackages.cuda_cudart}
          export LD_LIBRARY_PATH=${pkgs.lib.makeLibraryPath (cudaLibs ++ baseLibs)}:''${LD_LIBRARY_PATH:-}
          export CT2_FORCE_CPU=0

          echo "CUDA-enabled dev shell ready."
          echo "LD_LIBRARY_PATH is set so ctranslate2 GPU wheels can find libcudart, cuDNN, and cuBLAS."
        '';
      };
    };
}
