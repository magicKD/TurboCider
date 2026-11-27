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

## Web studio artwork

The web workspace uses a 96 px WebP copy of the production mark. Its PoseWear
contact sheet pairs the bundled demo inputs with actual outfit and pose edits
from the native Qwen Image 2.1 Turbo workflow. The transparency template uses
the lamp from the bundled PoseWear prop illustration with its background removed.

The built-in ImageGen tool produced the still life in
[amber-study.webp](../web-studio/amber-study.webp). The workspace ships a
1280 px WebP and lets the user try the scene as a prompt. The artwork prompt was

> 为 TurboCider 创作工作台拍一张横幅静物样片，画幅 4:3，无文字。暖橘色无缝摄影棚里，一只巨大的琥珀色竖纹玻璃圆盘斜倚在有棱角的浅杏色石台上，前方是一颗透明的小玻璃球。午后的硬光从左上方穿过玻璃，在台面投下锐利的条纹、焦散和长阴影。低机位，实物质感，紧凑而大胆的构图。高端设计杂志的静物摄影，轻微胶片颗粒。
