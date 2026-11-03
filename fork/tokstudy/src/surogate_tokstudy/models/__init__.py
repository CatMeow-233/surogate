"""The study's models on one PyTorch backend: Mamba-2 layers (mamba2), the main networks of the two tables (twin_a0
for tlc-N, trunk_bilingual for bilingual-B), the H-Net (hnet) and the token LMs (lm); `build` turns an arm into one
of them. Importing this package imports nothing; each module imports torch itself.
"""
