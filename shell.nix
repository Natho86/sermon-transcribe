{ pkgs ? import <nixpkgs> {} }:
pkgs.mkShell {
  packages = [
    pkgs.zlib
    pkgs.ffmpeg
    pkgs.pkg-config
  ];
}
