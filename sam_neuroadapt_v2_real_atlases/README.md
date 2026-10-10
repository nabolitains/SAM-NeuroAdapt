# `sam_neuroadapt_v2_real_atlases/` — model weights and base registration template

This folder is **empty in the repository**: nothing in it is redistributed. You fill it
yourself before running the pipeline. The folder name is fixed because the pipeline core
(`sam_neuroadapt_pipeline.ipynb`) looks for it at this exact path, relative to the
repository root.

```
sam_neuroadapt_v2_real_atlases/
├── models/    ← SAM2 checkpoint and configuration   (see models/README.md)
└── atlases/   ← MNI152NLin2009cAsym base template    (see atlases/README.md)
```

The pipeline also creates `data/`, `results/` and other working folders here at run time;
they are ignored by git.
