gbcamextract
============

Extracts photos from Game Boy Camera / Pocket Camera saves. Frames can be preserved. The Hello Kitty camera is supported too.

## Usage

```console
gbcamextract [-r rom.gb] -s save.sav
```

This will produce 30 AVIF files containing your photos. It is optional to specify the rom; this will allow the picture frames to be extracted too.

Each file is a lossless 8-bit monochrome AVIF still using
the four Game Boy grey levels (0, 85, 170, 255). Each file holds two
`av01` items in an `altr` group: the original-fidelity 160x144 raster
plus an 8x nearest-neighbor upscale (1280x1152) as the primary item, so
viewers display the large crisp version by default instead of
blurry-interpolating the tiny original. Both rasters are written by
a purpose-built lossless AV1 encoder (`src/mono.rs`): every 16x16 block is
coded `skip` with DC prediction plus a small luma palette holding the
block's exact colors, with adapting CDFs. On typical photos this lands
within a few percent of the old 2-bit PNG sizes (times ~64x the blocks
for the upscale). Layered output (border
as background, photo as overlay) is not used: an `iovl` overlay is not
valid AVIF, and AVIF grids require tiles of at least 64x64px, so neither
renders in standard viewers.

## Building

```console
cargo build --release
```

The binary is at `target/release/gbcamextract`. No system libraries are
needed: the container is written with the pure-Rust `gamut-isobmff`
crate, the arithmetic coder comes from pure-Rust `gamut-bitstream`, and
argument parsing uses `clap`.


## License

Licensed under the expat license, which is sometimes called the MIT license.
See the `LICENSE` file for details.
