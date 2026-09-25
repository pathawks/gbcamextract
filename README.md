gbcamextract
============

Extracts photos from Game Boy Camera / Pocket Camera saves. Frames can be preserved. The Hello Kitty camera is supported too.

## Usage

```console
gbcamextract [-r rom.gb] -s save.sav
```

This will produce 30 AVIF files containing your photos. It is optional to specify the rom; this will allow the picture frames to be extracted too.

Each file is a single lossless 160x144 8-bit AVIF still using the four
Game Boy grey levels (0, 85, 170, 255). Layered output (border as
background, photo as overlay) is not used: an `iovl` overlay is not valid
AVIF, and AVIF grids require tiles of at least 64x64px, so neither renders
in standard viewers.

## Building

```console
cargo build --release
```

The binary is at `target/release/gbcamextract`. No system libraries are
needed: AVIF encoding is done with the pure-Rust `gamut-avif` crate
(`gamut-av1` encoder, `gamut-isobmff` container), and argument parsing with
`clap`.


## License

Licensed under the expat license, which is sometimes called the MIT license.
See the `LICENSE` file for details.
