"""HarnessIR: an agentic harness for real-world image restoration.

The pipeline reads a degraded photograph, uses a vision-language model to
diagnose it and compose a restoration prompt, and hands that prompt to a
generative image editor together with the photograph.

Entry points, all runnable as scripts from the repository root:

    run_baseline.py   execute the prompt that ships in the manifest, no harness
    run_harness.py    the full harness: diagnosis -> tools -> weave -> execute
    compute_iqa.py    PSNR / SSIM / LPIPS / DISTS / MANIQA / CLIP-IQA / MUSIQ /
                      TOPIQ / AFINE-NR over a run's outputs
    compute_df.py     D-Score, F-Score and DF-Score over a run's outputs

See README.md for inputs, outputs and every parameter.
"""

__all__ = ["__version__"]
__version__ = "1.0.0"
