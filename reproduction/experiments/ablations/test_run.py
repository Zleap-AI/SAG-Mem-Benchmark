import asyncio
import unittest
from types import SimpleNamespace

from reproduction.experiments.ablations.run import evaluate


class FakeJob:
    key = "test/locomo/raw"
    run_id = "new"

    def __init__(self):
        self.value = {"question_id": "test", "run_id": "prepared", "reader_context": "raw body",
                      "question": "question", "target": {"answer": "answer", "category": "4"}}
        self.saved = []
        self.protocol = SimpleNamespace(reader_system="original prompt", score=lambda *a: {},
                        judge_prompt=lambda *a: "original judge prompt", judge_options=dict)

    def row(self, qid):
        return dict(self.saved[-1] if self.saved else self.value)

    def save(self, row):
        self.saved.append(dict(row))


class FakeReader:
    def __init__(self, fail_judge=False):
        self.readers = self.judges = self.active = self.peak = 0
        self.fail_judge = fail_judge

    async def respond(self, *args):
        self.readers += 1
        self.active += 1
        self.peak = max(self.active, self.peak)
        await asyncio.sleep(.001)
        self.active -= 1
        return SimpleNamespace(hypothesis="answer", prompt_tokens=10, completion_tokens=2, duration=.001)

    async def grade(self, *args, **kwargs):
        self.judges += 1
        if self.fail_judge:
            self.fail_judge = False
            raise ValueError("fixture judge failure")
        return SimpleNamespace(is_correct=True, prompt_tokens=8, completion_tokens=1)


class RunnerContracts(unittest.IsolatedAsyncioTestCase):
    async def test_reader_checkpoint_precedes_judge(self):
        job, reader = FakeJob(), FakeReader()
        await evaluate(job, "test", reader, asyncio.Semaphore(20))
        self.assertEqual(job.saved[0]["status"], "reader_complete")
        self.assertEqual(job.saved[-1]["status"], "succeeded")
        self.assertEqual(reader.readers, 1)
        self.assertEqual(reader.judges, 1)

    async def test_resume_failed_judge_does_not_repeat_reader(self):
        job, reader = FakeJob(), FakeReader(fail_judge=True)
        await evaluate(job, "test", reader, asyncio.Semaphore(20))
        self.assertEqual(job.saved[-1]["failure_stage"], "judge")
        await evaluate(job, "test", reader, asyncio.Semaphore(20))
        self.assertEqual(job.saved[-1]["status"], "succeeded")
        self.assertEqual(reader.readers, 1)
        self.assertEqual(reader.judges, 2)

    async def test_single_global_semaphore_caps_all_jobs(self):
        reader, semaphore = FakeReader(), asyncio.Semaphore(20)
        jobs = [FakeJob() for _ in range(45)]
        await asyncio.gather(*(evaluate(job, "test", reader, semaphore) for job in jobs))
        self.assertEqual(reader.peak, 20)
        self.assertTrue(all(job.saved[-1]["grade"] is True for job in jobs))


if __name__ == "__main__":
    unittest.main()
