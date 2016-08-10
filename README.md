gbcamextract
============

Extracts photos from Game Boy Camera saves. Frames are preserved. Requires [libpng](http://www.libpng.org/pub/png/libpng.html).

## Usage

```console
gbcamextract camerarom.gb save.sav
```

This will produce 30 PNG files containing your photos.

`black.gb` has been provided, which will output a black border for all photos, regardless of which border may have been originally set.


## Install

You will first need to install [libpng](http://www.libpng.org/pub/png/libpng.html)

Then, simply:
```console
make
```


## License

The MIT License (MIT)

Copyright (c) 2013 jkbenaim

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
