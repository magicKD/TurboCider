# Intermediate MLX command-buffer thresholds: completed screen

Code inspection confirms the pureGPU compiled path already avoids per-layer
host synchronization; the sampler synchronizes per diffusion step. Hybrid,
streaming, and eager compatibility branches retain intentional synchronization.
The existing1024MB experiment had not improved performance. Read Splash's
`runtime/metal/CommandGraph.hpp`: it owns an ordered list of dispatches for
submission, but does not establish that increasing MLX buffer limits helps.

No native source or library changes in this experiment. Same production
SHA256349630000153d6b294712f325c1e90b48fb8f11f5cf9048352d1d0c03f84b818,
BF16,pureGPU,512x512,8steps,greenhouse seed123. Four sequential fresh workers:
default-before, MLX_MAX_MB_PER_BUFFER=128, =256, default-after. Each has one
excluded cold request, two warm requests and one excluded parity request.
No concurrent GPU probe, driver/power change or system setting change.

See evolving raw reports (now complete) under
`/private/tmp/z-image-buffer-mid-screen`. Session97357 exited0.
First default warm diffusion14.738997833/14.884978667 s;128MB warm
15.153991750/14.952993125 s;256MB warm15.280993792/15.305997250 s;
final default14.818003541/14.881058917 s. Medians are14.811988250,
15.053492438,15.293495521,14.849531229 s respectively. This is a
four-arm bracketing screen, not a repeated pairwise ABBA qualification.
No candidate is promoted and default MLX limits remain untouched.

All four parity requests have identical eleven tensor files and PNG; actual
loaded runtime fingerprints match. This checks numerical compatibility for
the tested request, not a universal performance claim. The default execution
already avoids the suspected per-block waits; the changed thresholds did not
resolve512<10s or the historical absolute-runtime regression.
