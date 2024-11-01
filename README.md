Use this node like a RescaleCFG node, ... modelIn -> ThisNode -> ModelOut ... -> KSampler

See paper https://arxiv.org/pdf/2410.02416 for instructions about the other parameters. (Pages 20-21)

I have commented out the casting to .double() inside the function "project" as it was causing a parameter error
on my machine (DirectML can't use 'double', I believe). Shouldn't cause any noticeable difference, probably.

----------------------------------------------------------------------------------------------------------------------------------------------------

Edit4 - 01/11/2024:
Added an "adaptive_momentum". It gradually brings momentum towards 0 every step.

Adaptive_Momentum = 0 , No change to momentum

Adaptive_Momentum = 0.180 , Momentum will reach 0 roughly around the last step. I find that this helps to dramatically lower glitches/noise,
specially on lower steps or higher CFG values. This is now the default value.

Adaptive_Momentum = 0.190 , Momentum will reach 0 roughly around half the steps.

Added a "print_data" boolean to print on console the momentum at every step.

Issues:

Using denoise < 1.0 will start momentum at a lower value if adaptive_momentum > 0.

Using denoise < 1.0 will NOT reset momentum_buffer between gens (changing resolution or a value on the node will). May cause reproducibility issues.

The denoise < 1.0 problem exists since the 1st version. I haven't found a way to adjust for these yet.

----------------------------------------------------------------------------------------------------------------------------------------------------

Edit3 - 21/10/2024:
Now the Sampler's cfg is this node's previous "scale" value. May need to refresh the workflow and recreate the node.

----------------------------------------------------------------------------------------------------------------------------------------------------

Edit2:
Fixed a bug where momentum_buffer.running_average wouldn't reset between gens, changed defaults based on my tests again.

If you tried changing resolution without triggering a re-patch, you'd see a nice error message...

----------------------------------------------------------------------------------------------------------------------------------------------------

Edit:
Updated defaults, works well with high Norm_Threshold and higher scales


