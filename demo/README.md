# Streaming demo

The local demo starts one GPU server, streams generated text in the terminal,
and reports only prompt tokens, completion tokens, total tokens, server-side
decode TPS, and decode time. It stops the server and releases the GPU on
`/quit`, Ctrl+C, or an error.

Prerequisites are `uv`, an NVIDIA CUDA environment, and at least one compatible
checkpoint under `models/`.

Choose `active.model`, `active.config`, and `server.gpu` in `config.yaml`, then
run:

```bash
./demo/run.sh
```

The first invocation creates the locked local environment, and later
invocations reuse it. Enter prompts interactively and use `/quit` to stop the
server or `/clear` to clear the terminal. A one-shot prompt is also supported:

```bash
./demo/run.sh --prompt "Explain recurrent language models."
```

The model catalog mirrors the baseline and LoopSpec configurations in
`experiments/*.yaml`. LoopSpec remains single-GPU and always streams with interval 1.
Decode time uses the same window as SGLang's decode TPS and excludes the
first token: `(completion tokens - 1) / decode TPS`.
