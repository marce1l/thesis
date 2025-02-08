let
  pkgs =
    import
      (fetchTarball "https://github.com/NixOS/nixpkgs/archive/1da52dd49a127ad74486b135898da2cef8c62665.tar.gz")
      { };
in
pkgs.mkShell {
  packages = [
    (pkgs.python3.withPackages (
      python-pkgs: with python-pkgs; [
        numpy
        matplotlib
      ]
    ))
  ];
}
