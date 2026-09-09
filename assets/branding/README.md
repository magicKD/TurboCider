# Turbo Drop branding

`TurboDrop.png` is the production artwork derived from the user-supplied
TurboCider concept on 2026-09-09 using the built-in ImageGen tool.

The edit prompt was: extract the central droplet, preserve its silhouette,
sweeping negative-space cut, cyan/turquoise-to-lime gradient and pale inner
crescent; center a single large mark on dark navy with padding; remove the
concept captions, wordmark and small icon mockup, without redesigning it.

Run `swift tools/branding/generate.swift` from the repository root to regenerate
`LogoMark.png`, `AppIcon.icns`, `mark.svg` and `logo.svg`. The SVG containers
embed the PNG artwork; they are not independent vector reconstructions.
The source PNG is checked in, so regeneration needs no network or image model.
