# shell.nix
with import <nixpkgs> {};

mkShell rec {
  buildInputs = [
    python310  # Add the Python version you need
    python310Packages.virtualenv  # Ensure you have virtualenv available
    cmake
  ];

  # Setting the LD_LIBRARY_PATH, if necessary, for additional libraries
  NIX_LD_LIBRARY_PATH = lib.makeLibraryPath [
    stdenv.cc.cc
    zlib
  ];
  LD_LIBRARY_PATH = NIX_LD_LIBRARY_PATH;
  NIX_LD = lib.fileContents "${stdenv.cc}/nix-support/dynamic-linker";

  shellHook = ''
    # Activate the virtual environment
    if [ -d ".venv" ]; then
      source .venv/bin/activate
    else
      echo "No virtual environment found, create it using python -m venv .venv"
    fi
  '';
}

