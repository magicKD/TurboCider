# H3 runtime provenance

This directory is the active, self-contained H3 C/Objective-C/Metal runtime
used by TurboCider. It was derived from the sibling `h3.c` repository at base
commit `fceeb88122c44e42785956b320f0753c339978ef` and adapted for the
TurboCider library configuration, resource discovery, callbacks and native
media handoff. The applicable MIT notices are included in
`native/THIRD_PARTY_NOTICES.md`.

Model math remains owned by this runtime. `h3_session.mm` should contain only
TurboCider request validation, manifest/LoRA identity, lifecycle, cancellation
and result translation; do not create a second H3 operator implementation in
the Session layer.

The sibling repository may contain later uncommitted experiments. This
vendored directory is the source compiled into TurboCider and must be reviewed
and tested as its own snapshot before release.
