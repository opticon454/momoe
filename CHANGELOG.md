# Changelog

## 20261110-1529

- Kept mailbox polling, fences and timeouts on pre-SM70 GPUs while avoiding the unsupported nanosleep instruction; added bounded mapped-memory wakeup checks.

## 20261110-1517

- Added an opt-in CUDA 12 Maxwell SM52 build in a separate output directory, with a strict toolkit check and bounded model-free CUDA matrix validation.
