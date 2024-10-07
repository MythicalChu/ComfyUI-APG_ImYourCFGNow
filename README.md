Use this node like a RescaleCFG node, ... modelIn -> ThisNode -> ModelOut ... -> KSampler

"scale" acts like your CFG, your CFG doesn't do anything anymore white this node is active.
See paper https://arxiv.org/pdf/2410.02416 for instructions about the other parameters. (Pages 20-21)

I have commented out the casting to .double() inside the function "project" as it was causing a parameter error
on my machine (DirectML can't use 'double', I believe). Shouldn't cause any noticeable difference, probably.

Edit:
Updated defaults, works well with high Norm_Threshold and higher scales

Edit2:
Fixed a bug where momentum_buffer.running_average wouldn't reset between gens, changed defaults based on my tests again.
If you tried changing resolution without triggering a re-patch, you'd see a nice error message...
