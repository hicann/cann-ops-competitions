from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class SourceContractTest(unittest.TestCase):
    def test_host_acquires_one_full_connect_channel_per_peer(self):
        source = (ROOT / "op_host" / "reduce_scatter.cc").read_text(encoding="utf-8")
        for token in (
            "FULL_CONNECT_NET_LAYER = 1",
            "HcclRankGraphGetLinks",
            "HcclChannelDescInit",
            "HcclChannelAcquire",
            "HcclChannelGetHcclBuffer",
            "AcquireChannels(comm, param, resCtxHost)",
        ):
            self.assertIn(token, source)

    def test_device_uses_chunked_deterministic_xor_reduction(self):
        source = (ROOT / "op_kernel_aicpu" / "exec_op.cc").read_text(encoding="utf-8")
        for token in (
            "MAX_DATA_SIZE = 256ULL * 1024 * 1024",
            "param.myRank ^ roundId",
            "HcommWriteOnThread",
            "HcommLocalReduceOnThread",
            "HcommChannelNotifyRecordOnThread",
            "HcommChannelNotifyWaitOnThread",
        ):
            self.assertIn(token, source)
        self.assertNotIn("算法任务编排", source)

    def test_large_path_overlaps_output_copy_before_worker_join(self):
        source = (ROOT / "op_kernel_aicpu" / "exec_op.cc").read_text(encoding="utf-8")
        copy_token = "CopyBytesOnThread(crossThread, output, crossPartial, totalBytes)"
        join_token = "HcommThreadNotifyRecordOnThread(crossThread, localThread, 0)"
        final_reduce_token = "output, localPartial, param.count, hcommDataType, hcommReduceOp"
        self.assertIn(copy_token, source)
        self.assertIn(join_token, source)
        self.assertIn(final_reduce_token, source)
        self.assertLess(source.index(copy_token), source.index(join_token))


if __name__ == "__main__":
    unittest.main()
