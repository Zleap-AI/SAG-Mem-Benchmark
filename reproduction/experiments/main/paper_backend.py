"""SAG-Mem (§3.1–§3.4): paper chunk boundaries and retained source metadata over the SAG2 backend."""
from pathlib import Path
from shutil import copy2

from reproduction.experiments.shared import chunking
from reproduction.experiments.shared.chunk_width import split_options as _messages_and_options
from src.backends import sag2_adapter
from src.backends.sag2_chunk_time import SAG2ChunkTimeBackend
from src.harness.benchmark import file_digest, register


class PaperSAGMem(SAG2ChunkTimeBackend):
    def __init__(self, options):
        messages_per_chunk, sag_options = _messages_and_options(options)
        super().__init__(sag_options)
        self.messages_per_chunk = messages_per_chunk
        if messages_per_chunk is not None:
            backend_options = {**self.options.get("backend_options", {}),
                               "messages_per_chunk": messages_per_chunk}
            self.options = {**self.options, "backend_options": backend_options}

    @classmethod
    def validate(cls, options, examples):
        _, sag_options = _messages_and_options(options)
        super().validate(sag_options, examples)

    @classmethod
    def dependencies(cls, options):
        messages_per_chunk, sag_options = _messages_and_options(options)
        deps = super().dependencies(sag_options)
        published = {**deps["chunking"], "policy": chunking.POLICY,
                     "messages_per_chunk": None, "budget": "soft",
                     "oversized_message": "standalone_full_text",
                     "implementation_sha256": file_digest(Path(chunking.__file__))}
        if messages_per_chunk is None:
            deps["chunking"] = published
            return deps
        from reproduction.experiments.shared import chunk_width
        deps["chunking"] = {**published, "policy": f"{messages_per_chunk}_messages_soft512_atomic_v1",
                            "messages_per_chunk": messages_per_chunk,
                            "implementation_sha256": file_digest(Path(chunk_width.__file__))}
        return deps

    @classmethod
    def runtime_dependencies(cls, options):
        _, sag_options = _messages_and_options(options)
        return super().runtime_dependencies(sag_options)

    async def prepare(self, history, build_spec):
        messages_per_chunk = self.messages_per_chunk
        original = sag2_adapter.chunk_history
        if messages_per_chunk is None:
            sag2_adapter.chunk_history = chunking.chunk_history
            policy = chunking.POLICY
        else:
            from reproduction.experiments.shared.chunk_width import chunk_history as chunk_width
            sag2_adapter.chunk_history = lambda item: chunk_width(item, messages_per_chunk)
            policy = f"{messages_per_chunk}_messages_soft512_atomic_v1"
        try:
            await super().prepare(history, build_spec)
            self.build_stats.update(messages_per_chunk=messages_per_chunk, chunk_policy=policy)
        finally:
            sag2_adapter.chunk_history = original
        root = self.options.get("retained_sources_root", Path(__file__).parent / "retained_sources")
        retained = Path(root) / build_spec.id
        retained.mkdir(parents=True, exist_ok=True)
        for filename in ("build.json", "sources.json"):
            copy2(build_spec.directory / filename, retained / filename)


register("backend", "sag-mem", PaperSAGMem)
