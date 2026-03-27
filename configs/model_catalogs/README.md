# Model Catalog Scaffold

This directory is reserved for the next-stage model management catalog.

The goal is to keep catalog data separate from runtime code so the download
orchestrator can resolve:

- storage root
- visibility
- source kind
- auth requirements
- filename / relative path

The current notebook pattern maps well to this structure:

- a top-level catalog file
- grouped entries by family or usage
- a small number of storage roots
- explicit support assets that do not mix into the generic LoRA pool

Planned next-mission consumers:

- `modules/model_download/catalog.py`
- `modules/model_download/policy.py`
- `modules/model_download/resolver.py`
- `modules/model_download/transport.py`
- `modules/model_download/orchestrator.py`
