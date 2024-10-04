Use this node like a RescaleCFG node, ... modelIn -> ThisNode -> ModelOut ... -> KSampler

"scale" acts like your CFG, your CFG doesn't do anything anymore white this node is active.
See paper https://arxiv.org/pdf/2410.02416 for instructions about the other parameters. (Pages 20-21)

I'm not entirely sure I've implemented it correctly or the best way, but it seems to work fine.
I've only tested it on SDXL, 5.0 scale seems similar to 7.0 CFG

I have commented out the casting to .double() inside the function "project" as it was causing a parameter error
on my machine (DirectML can't use 'double', I believe). Shouldn't cause any noticeable difference, probably.
