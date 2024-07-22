{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-23.11";
    nixpkgs_master.url = "github:NixOS/nixpkgs/master";
    systems.url = "github:nix-systems/default";
    devenv.url = "github:cachix/devenv";
  };

  nixConfig = {
    extra-trusted-public-keys = "devenv.cachix.org-1:w1cLUi8dv3hnoSPGAuibQv+f9TZLr6cv/Hm9XgU50cw=";
    extra-substituters = "https://devenv.cachix.org";
  };

  outputs = { self, nixpkgs, devenv, systems, ... } @ inputs:
    let
      forEachSystem = nixpkgs.lib.genAttrs (import systems);
    in
    {
      packages = forEachSystem (system: {
        devenv-up = self.devShells.${system}.default.config.procfileScript;
      });

      devShells = forEachSystem
        (system:
          let
            pkgs = import nixpkgs {
              system = system;
              config.allowUnfree = true;
            };

            frameworks = pkgs.darwin.apple_sdk.frameworks;
            mpkgs = import inputs.nixpkgs_master {
              system = system;
              config.allowUnfree = true;
            };

            libList = [
                # Add needed packages here
                pkgs.stdenv.cc.cc
                pkgs.libGL
                pkgs.glib
              ] ++ pkgs.lib.optionals pkgs.stdenv.isLinux [
                pkgs.cudaPackages.cudatoolkit
                pkgs.cudaPackages.libcublas
                pkgs.cudaPackages.libcurand
                pkgs.cudaPackages.cudnn
                pkgs.cudaPackages.libcufft
                pkgs.linuxPackages.nvidia_x11
              ];
          in
          {
            default = devenv.lib.mkShell {
              inherit inputs pkgs;
              modules = [
                {
                  env.NIX_LD = nixpkgs.lib.fileContents "${pkgs.stdenv.cc}/nix-support/dynamic-linker";
                  env.NIX_LD_LIBRARY_PATH = nixpkgs.lib.makeLibraryPath libList;
                  packages = with pkgs; [
                    micromamba
                    poetry
                  ]; 
                  enterShell = ''
                    
		    export LD_LIBRARY_PATH=$NIX_LD_LIBRARY_PATH
                    export SSL_CERT_FILE="${pkgs.cacert}/etc/ssl/certs/ca-bundle.crt"
		    export PYTHON_KEYRING_BACKEND=keyring.backends.fail.Keyring
                    eval "$(micromamba shell hook -s bash)"
                    if [ ! -d ".venv/envs/counterfactuals" ]; then
                       micromamba create -r .venv -n counterfactuals python=3.10 ipykernel pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -c conda-forge -y
                       micromamba activate .venv/envs/counterfactuals 
                       python -m ipykernel install --user --name counterfactuals
                       poetry install -vv --with dev --no-root 
                    fi
                    micromamba activate .venv/envs/counterfactuals
                  '';
                }
              ];
            };
          });
    };
}