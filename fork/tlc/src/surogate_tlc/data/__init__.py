"""TLC data: the corpus spec (corpus), decontamination (decontam), the synthetic binding programs (synth), probe
items and their floors (items, floors, natural), the tokenizer (tokenizer) and the token shards the native trainer
reads (shards).

None of these modules needs the native extension; the tokenizer check loads it only when it runs against the native
tokenizer. Each tool that writes files is a module entry point: python -m surogate_tlc.data.<module>.
"""
