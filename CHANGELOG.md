# Changelog

## 20261110-1610

- Recorded the validated Maxwell SM52 results in the docs: bounded checker, one model case and the serve-protocol handshake on an actual Quadro M4000, with the numbers and what they do not claim.

## 20261110-1529

- Kept mailbox polling, fences and timeouts on pre-SM70 GPUs while avoiding the unsupported nanosleep instruction; added bounded mapped-memory wakeup checks.

## 20261110-1517

- Added an opt-in CUDA 12 Maxwell SM52 build in a separate output directory, with a strict toolkit check and bounded model-free CUDA matrix validation.
