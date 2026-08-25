# outputs

Everything the scripts compute lands here. Nothing in this directory is source,
and it can be deleted in full and rebuilt.

    checkpoint/               the trained model, 154 angles
    paper_numbers.json        every computed quantity
    numbers.tex               the same values as LaTeX macros, input by the paper
    manuscript_reference.tex  the macro table as printed in the submitted paper,
                              which scripts/reproduce.py checks against
    score/                    held-out scores with confidence intervals
    bell_generative.*         the measurement-setting certification result
