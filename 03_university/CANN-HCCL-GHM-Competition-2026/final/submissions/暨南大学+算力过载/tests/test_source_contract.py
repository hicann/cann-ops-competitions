from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[1]
MAX_LOCAL_COPY_BYTES = 256 * 1024 * 1024
MAIN_CHUNK_ALIGN = 128


def read(path: str) -> str:
    target = ROOT / path
    return target.read_text(encoding="utf-8") if target.is_file() else ""


def bounded_local_copy_chunks(total_bytes: int) -> list[int]:
    if total_bytes < 0:
        raise ValueError("copy size must be non-negative")
    chunks = []
    remaining = total_bytes
    while remaining:
        current = min(MAX_LOCAL_COPY_BYTES, remaining)
        chunks.append(current)
        remaining -= current
    return chunks


def reduction_aware_main_chunk(rank_size: int, workspace_bytes: int) -> int:
    if rank_size < 2:
        raise ValueError("reduction chunking requires at least two ranks")
    max_reduce_pieces = max(1, rank_size // 2)
    reduction_limit = MAX_LOCAL_COPY_BYTES // max_reduce_pieces
    main = min(MAX_LOCAL_COPY_BYTES, workspace_bytes // rank_size, reduction_limit)
    return main - main % MAIN_CHUNK_ALIGN


class FinalSourceContractTest(unittest.TestCase):
    def test_all_six_editable_files_exist(self):
        for path in (
            "include/custom.h", "op_host/reduce_scatter.cc", "op_host/exec_op.h",
            "op_host/exec_op.cc", "op_kernel_ccu/ccu_kernel.h", "op_kernel_ccu/ccu_kernel.cc",
        ):
            self.assertTrue((ROOT / path).is_file(), path)

    def test_kernel_stages_then_reduces_in_fixed_order(self):
        source = read("op_kernel_ccu/ccu_kernel.cc")
        for token in (
            "INPUT_ADDR_XN_ID", "INPUT_TOKEN_XN_ID", "PRE_SYNC_NOTIFY_IDX",
            "POST_SYNC_BIT", "TOTAL_TASK_ARGS = 9", "WriteVariableWithNotify",
            "NotifyWait", "ccu::Read", "ccu::LocalCopy", "ccu::EventWait",
            "ccu::LocalReduce", "remainingPieces", "reducePieces",
        ):
            self.assertIn(token, source)
        self.assertNotIn("ccu::WriteReduce", source)
        self.assertNotIn("ccu::ReadReduce", source)

    def test_tokens_are_not_written_to_formatted_logs(self):
        combined = "\n".join(read(path) for path in (
            "op_host/reduce_scatter.cc", "op_host/exec_op.cc", "op_kernel_ccu/ccu_kernel.cc",
        ))
        self.assertIsNone(re.search(
            r"(?:HCCL_(?:INFO|DEBUG|WARNING|ERROR)|printf)\s*\([^\n;]*token[^\n;]*%",
            combined,
            re.IGNORECASE,
        ))

    def test_shared_headers_define_final_ccu_interfaces(self):
        custom = read("include/custom.h")
        kernel = read("op_kernel_ccu/ccu_kernel.h")
        for token in ("CcuKernelArgBase", "CcuKernelInfo", "AlgResourceCtx"):
            self.assertIn(token, custom)
        for token in ("ReduceScatterKernelArg", "CcuReduceScatterKernel"):
            self.assertIn(token, kernel)

    def test_custom_header_reuses_online_baseline_symbols(self):
        custom = read("include/custom.h")
        self.assertIn('#include "binary_stream.h"', custom)
        self.assertIn('#include "common.h"', custom)
        self.assertIsNone(re.search(
            r"constexpr\s+\w+\s+MAX_RANK_SIZE\s*=",
            custom,
        ))
        self.assertIsNone(re.search(
            r"inline\s+HcclResult\s+ConvertCcuToHccl\s*\(",
            custom,
        ))

    def test_resource_context_serialization_is_inline_and_field_order_is_stable(self):
        custom = read("include/custom.h")
        serialize = re.search(
            r"std::vector<char> Serialize\(\)\s*\{(?P<body>.*?)\n    \}", custom, re.DOTALL
        )
        deserialize = re.search(
            r"void DeSerialize\(std::vector<char> &data\)\s*\{(?P<body>.*?)\n    \}", custom, re.DOTALL
        )
        self.assertIsNotNone(serialize)
        self.assertIsNotNone(deserialize)
        serialize_body = serialize.group("body")
        deserialize_body = deserialize.group("body")
        serialized_fields = ["ccuThread", "localBuffer", "threads", "ccuKernels"]
        self.assertEqual(
            [match.group(1) for match in re.finditer(r"binaryStream << (\w+);", serialize_body)],
            serialized_fields,
        )
        self.assertEqual(
            [match.group(1) for match in re.finditer(r"binaryStream >> (\w+);", deserialize_body)],
            serialized_fields,
        )

    def test_control_plane_uses_ccu_and_one_channel_per_peer(self):
        source = read("op_host/reduce_scatter.cc")
        for token in (
            "COMM_ENGINE_CCU", "HcclRankGraphGetLayers", "HcclRankGraphGetLinks",
            "COMM_PROTOCOL_UBC_CTP", "HcclChannelAcquire", "HcclCommQueryCcuIns",
            "HcommCcuKernelRegisterStart", "HcommCcuKernelRegister",
            "HcommCcuKernelRegisterEnd", "param.reduceType = op",
        ):
            self.assertIn(token, source)
        self.assertNotIn("COMM_ENGINE_AICPU_TS", source)

    def test_host_registers_dfx_before_ccu_work(self):
        source = read("op_host/reduce_scatter.cc")
        get_name = "CHK_RET(HcclGetCommName(comm, commName));"
        register = (
            "CHK_RET(HcclDfxRegOpInfoByCommId(commName, "
            "reinterpret_cast<void *>(&dfxInfo)));"
        )
        self.assertIn("HcclDfxOpInfo dfxInfo;", source)
        self.assertIn("char commName[COMM_INDENTIFIER_MAX_LENGTH];", source)
        self.assertIn(get_name, source)
        self.assertIn(register, source)
        self.assertLess(source.index(get_name), source.index("HcclGetRankId"))
        self.assertLess(source.index(register), source.index("HcclGetRankId"))
        self.assertLess(source.index(register), source.index("HcclThreadAcquireWithStream"))
        self.assertLess(source.index(register), source.index("return ops_hccl::ExecOp(param);"))

    def test_cached_context_refreshes_the_current_stream_thread(self):
        source = read("op_host/reduce_scatter.cc")
        self.assertIn("resCtxHost.threads[0] = param.cpuThread", source)

    def test_context_lookup_accepts_legacy_and_v2_misses_but_propagates_other_errors(self):
        source = read("op_host/reduce_scatter.cc")
        self.assertIn("ctxRet == HCCL_E_PARA || ctxRet == HCCL_E_NOT_FOUND", source)
        self.assertIn("CHK_PRT_RET(!contextMissing", source)
        self.assertIn("ctxRet);", source)

    def test_failed_initial_context_copy_destroys_the_published_context(self):
        source = read("op_host/reduce_scatter.cc")
        self.assertIn("HcclEngineCtxDestroy", source)

    def test_dispatch_chunks_and_launches_nine_arguments(self):
        source = read("op_host/exec_op.cc")
        for token in (
            "MAX_DATA_SIZE", "SafeMultiply", "AlignDown", "HcommCcuGetMemToken",
            "HcommCcuKernelLaunch", "localBuffer.size / param.rankSize",
            "inputSliceOffset", "outputOffset", "TASK_ARG_COUNT = 9",
        ):
            self.assertIn(token, source)

    def test_rank_one_copy_model_obeys_the_operation_limit_and_exact_tail(self):
        self.assertEqual(bounded_local_copy_chunks(0), [])
        self.assertEqual(bounded_local_copy_chunks(4), [4])
        self.assertEqual(bounded_local_copy_chunks(MAX_LOCAL_COPY_BYTES), [MAX_LOCAL_COPY_BYTES])
        self.assertEqual(
            bounded_local_copy_chunks(2 * MAX_LOCAL_COPY_BYTES + 4),
            [MAX_LOCAL_COPY_BYTES, MAX_LOCAL_COPY_BYTES, 4],
        )

    def test_rank_one_dispatch_uses_bounded_local_copy_chunks(self):
        source = read("op_host/exec_op.cc")
        for token in (
            "CopyBytesOnThread", "std::min(MAX_DATA_SIZE, bytes - offset)",
            "dstBytes + offset", "srcBytes + offset",
        ):
            self.assertIn(token, source)
        rank_one = re.search(r"if \(param\.rankSize == 1\) \{(?P<body>.*?)\n    \}", source, re.DOTALL)
        self.assertIsNotNone(rank_one)
        self.assertIn("CopyBytesOnThread", rank_one.group("body"))
        self.assertNotIn("HcommLocalCopyOnThread", rank_one.group("body"))

    def test_reduction_aware_chunk_model_caps_the_largest_tree_operation(self):
        large_workspace = 16 * MAX_LOCAL_COPY_BYTES
        expected_main_chunks = {
            2: 268435456,
            4: 134217728,
            12: 44739200,
            16: 33554432,
        }
        for rank_size, expected in expected_main_chunks.items():
            self.assertEqual(reduction_aware_main_chunk(rank_size, large_workspace), expected)
        for rank_size in range(2, 17):
            chunk = reduction_aware_main_chunk(rank_size, large_workspace)
            self.assertLessEqual(chunk, MAX_LOCAL_COPY_BYTES)
            self.assertLessEqual((rank_size // 2) * chunk, MAX_LOCAL_COPY_BYTES)
            self.assertEqual(chunk % MAIN_CHUNK_ALIGN, 0)

    def test_dispatch_caps_chunks_for_the_largest_local_reduce(self):
        source = read("op_host/exec_op.cc")
        for token in (
            "maxReducePieces", "param.rankSize / 2", "reductionChunkLimit",
            "MAX_DATA_SIZE / maxReducePieces",
        ):
            self.assertIn(token, source)
        main_chunk = re.search(
            r"uint64_t mainChunk\s*=\s*std::min<uint64_t>\((?P<body>.*?)\);",
            source,
            re.DOTALL,
        )
        self.assertIsNotNone(main_chunk)
        self.assertIn("reductionChunkLimit", main_chunk.group("body"))
        self.assertIn("localBuffer.size / param.rankSize", main_chunk.group("body"))


if __name__ == "__main__":
    unittest.main()
