# Third-party research references

## maderix/ANE

`Sources/PrivateANEProbe/main.mm` adapts the in-memory MIL compilation and
IOSurface request mechanism demonstrated by
[maderix/ANE](https://github.com/maderix/ANE), commit
`d91c9845c0784dec7753048954fc6d0e8411fe29`.

The local probe materially extends that mechanism with dynamic weight payload
length and explicit payload-offset serialization, arbitrary rectangular
projection shapes, file-backed real weights, INT8 boundary I/O, IOSurface to
Metal `bytesNoCopy` consumption, and differential numerical checks. In
particular, it does not retain the upstream benchmark's 512×512-specific
one-byte payload-length shortcut.

MIT License

Copyright (c) 2026 maderix

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
