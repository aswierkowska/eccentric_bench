with import <nixpkgs> {};

let
  pythonEnv = python3.withPackages (ps: with ps; [ pip virtualenv ]);

  # Define it here, outside mkShell attribute set:
  nixLdLibraryPath = lib.makeLibraryPath [
    stdenv.cc.cc
    zlib
  ];
in

mkShell {
  buildInputs = [
    pythonEnv
    cmake
  ];

  # Now just use the let-bound variable:
  NIX_LD_LIBRARY_PATH = nixLdLibraryPath;
  LD_LIBRARY_PATH = nixLdLibraryPath;

  NIX_LD = builtins.readFile "${stdenv.cc}/nix-support/dynamic-linker";

  shellHook = ''
    if [ -d ".venv" ]; then
      source .venv/bin/activate
    else
      echo "No virtual environment found, create it using python -m venv .venv"
    fi
  '';
}

