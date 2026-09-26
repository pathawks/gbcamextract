gbcamextract
============

Extracts photos from Game Boy Camera / Pocket Camera saves. Frames can be preserved. The Hello Kitty camera is supported too.

## Usage

```console
gbcamextract [-r rom.gb] -s save.sav
```

This will produce 30 PNG files containing your photos. It is optional to specify the rom; this will allow the picture frames to be extracted too.

## Building (Rust)

```console
cargo build --release
```

The binary is at `target/release/gbcamextract`. No system libraries are
needed: PNG encoding is done with the pure-Rust `png` crate, and argument
parsing with `clap`.

## Building (legacy C)

You will first need to install [libpng](http://www.libpng.org/pub/png/libpng.html).

Then, for Linux:
```console
make
```

Or for Windows:
```console
mingw32-make -f Makefile.win
```


## License

Licensed under the expat license, which is sometimes called the MIT license.
See the `LICENSE` file for details.
