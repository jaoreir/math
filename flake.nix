{
  description = "Flake for python development.";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs?ref=nixos-unstable";
  };

  outputs = { self, nixpkgs }:
    let
      system = "x86_64-linux";
    in
    {
      devShells.${system}.default =
        let
          pkgs = nixpkgs.legacyPackages.${system};
          python = pkgs.python310;
          pythonPackages = python.pkgs;
          lib-path = with pkgs; lib.makeLibraryPath [
            # These are needed for numpy
            stdenv.cc.cc
            libz

            # Needed for xorg
            xorg.libX11

            # These are allegedly needed for wayland
            # glib
            # zlib
            # libGL
            # fontconfig
            # libxkbcommon
            # freetype
          ];
        in
        with pkgs; mkShell {
          packages = with pythonPackages;[
            venvShellHook
            tkinter
          ];

          buildInputs = [
            git
            python
            readline
          ];

          shellHook = ''
            PYTHON_NAME=python3.10
            SOURCE_DATE_EPOCH=$(date +%s)
            export "LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${lib-path}"
            VENV=venv

            if test ! -d $VENV; then
              $PYTHON_NAME -m venv $VENV
            fi
            source ./$VENV/bin/activate
            export PYTHONPATH=`pwd`/$VENV/${python.sitePackages}/:$PYTHONPATH
            pip install -r requirements.txt
          '';

          postShellHook = ''
            ln -sf ${python.sitePackages}/* ./venv/lib/$PYTHON_NAME/site-packages
          '';
        };
    };
}
