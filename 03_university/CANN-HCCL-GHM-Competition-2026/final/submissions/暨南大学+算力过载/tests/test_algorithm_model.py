import random
import struct
import unittest


MAX_OP_BYTES = 256 * 1024 * 1024
WORKSPACE_BYTES = 400 * 1024 * 1024
MAIN_ALIGN = 128


def chunk_sizes(total_bytes: int, rank_size: int, workspace_bytes: int = WORKSPACE_BYTES) -> list[int]:
    if total_bytes < 0 or total_bytes % 4:
        raise ValueError("FP32 byte size must be non-negative and four-byte aligned")
    if not 1 <= rank_size <= 16:
        raise ValueError("rank_size must be in [1, 16]")
    if total_bytes == 0:
        return []
    workspace_limit = workspace_bytes // rank_size
    max_reduce_pieces = max(1, rank_size // 2)
    reduction_limit = MAX_OP_BYTES // max_reduce_pieces
    main = min(MAX_OP_BYTES, workspace_limit, reduction_limit)
    main -= main % MAIN_ALIGN
    if main < 4:
        raise ValueError("workspace cannot hold one FP32 value per rank")
    chunks = []
    remaining = total_bytes
    while remaining:
        current = min(main, remaining)
        chunks.append(current)
        remaining -= current
    return chunks


def tree_rounds(rank_size: int) -> list[tuple[int, int, int]]:
    if not 1 <= rank_size <= 16:
        raise ValueError("rank_size must be in [1, 16]")
    rounds = []
    remaining = rank_size
    while remaining > 1:
        pieces = remaining // 2
        rounds.append((0, remaining - pieces, pieces))
        remaining -= pieces
    return rounds


def fp32(value: float) -> float:
    return struct.unpack("f", struct.pack("f", value))[0]


def modeled_reduce(values: list[float]) -> float:
    work = [fp32(value) for value in values]
    remaining = len(work)
    while remaining > 1:
        pieces = remaining // 2
        source = remaining - pieces
        for index in range(pieces):
            work[index] = fp32(work[index] + work[source + index])
        remaining -= pieces
    return work[0]


class FinalChunkPlanTest(unittest.TestCase):
    def test_all_scored_shapes_obey_operation_and_workspace_limits(self):
        for rank_size in (4, 12, 16):
            for total_bytes in (512 * 1024, 512 * 1024 * 1024, 400 * 1024 * 1024 + 4):
                chunks = chunk_sizes(total_bytes, rank_size)
                self.assertEqual(sum(chunks), total_bytes)
                self.assertTrue(all(chunk <= MAX_OP_BYTES for chunk in chunks))
                self.assertTrue(all(chunk * rank_size <= WORKSPACE_BYTES for chunk in chunks))
                self.assertTrue(all(chunk % 4 == 0 for chunk in chunks))

    def test_four_rank_400mb_plus_four_has_exact_four_byte_tail(self):
        chunks = chunk_sizes(400 * 1024 * 1024 + 4, 4)
        self.assertEqual(chunks, [100 * 1024 * 1024] * 4 + [4])

    def test_every_tree_round_obeys_the_operation_limit(self):
        large_workspace = 16 * MAX_OP_BYTES
        total_bytes = 3 * MAX_OP_BYTES + 4
        expected_main_chunks = {
            2: 268435456, 3: 268435456,
            4: 134217728, 5: 134217728,
            6: 89478400, 7: 89478400,
            8: 67108864, 9: 67108864,
            10: 53687040, 11: 53687040,
            12: 44739200, 13: 44739200,
            14: 38347904, 15: 38347904,
            16: 33554432,
        }
        for rank_size in range(2, 17):
            chunks = chunk_sizes(total_bytes, rank_size, workspace_bytes=large_workspace)
            self.assertEqual(chunks[0], expected_main_chunks[rank_size])
            for _, _, reduce_pieces in tree_rounds(rank_size):
                self.assertTrue(all(reduce_pieces * chunk <= MAX_OP_BYTES for chunk in chunks))

    def test_workspace_dominant_plan_has_exact_four_byte_tail(self):
        chunks = chunk_sizes(WORKSPACE_BYTES + 4, 16)
        self.assertEqual(chunks, [25 * 1024 * 1024] * 16 + [4])

    def test_zero_bytes_has_no_launches(self):
        self.assertEqual(chunk_sizes(0, 16), [])

    def test_invalid_rank_sizes_and_tiny_workspace_are_rejected(self):
        for rank_size in (0, 17):
            with self.assertRaises(ValueError):
                chunk_sizes(4, rank_size)
        with self.assertRaises(ValueError):
            chunk_sizes(4, 16, workspace_bytes=63)

    def test_single_rank_obeys_exact_256mb_operation_limit(self):
        chunks = chunk_sizes(512 * 1024 * 1024, 1, workspace_bytes=1024 * 1024 * 1024)
        self.assertEqual(chunks, [MAX_OP_BYTES, MAX_OP_BYTES])


class FinalReductionTreeTest(unittest.TestCase):
    def test_expected_tree_shapes(self):
        self.assertEqual([r[2] for r in tree_rounds(4)], [2, 1])
        self.assertEqual([r[2] for r in tree_rounds(12)], [6, 3, 1, 1])
        self.assertEqual([r[2] for r in tree_rounds(16)], [8, 4, 2, 1])

    def test_reduction_is_bitwise_deterministic(self):
        values = [1e20, 1.0, -1e20, 3.0, 7.0, -2.0, 0.25, 0.5,
                  11.0, -13.0, 17.0, 19.0, -23.0, 29.0, 31.0, -37.0]
        for rank_size in (4, 12, 16):
            first = struct.pack("f", modeled_reduce(values[:rank_size]))
            for _ in range(20):
                self.assertEqual(struct.pack("f", modeled_reduce(values[:rank_size])), first)

    def test_seeded_random_fp32_reduction_is_bitwise_deterministic(self):
        random_source = random.Random(20260727)
        values = [fp32(random_source.uniform(-1e20, 1e20)) for _ in range(16)]
        for rank_size in range(1, 17):
            first = struct.pack("f", modeled_reduce(values[:rank_size]))
            for _ in range(20):
                self.assertEqual(struct.pack("f", modeled_reduce(values[:rank_size])), first)

    def test_tree_lineage_contains_every_rank_exactly_once(self):
        for rank_size in range(1, 17):
            lineages = [{rank} for rank in range(rank_size)]
            for destination, source, reduce_pieces in tree_rounds(rank_size):
                for index in range(reduce_pieces):
                    destination_lineage = lineages[destination + index]
                    source_lineage = lineages[source + index]
                    self.assertTrue(destination_lineage.isdisjoint(source_lineage))
                    destination_lineage.update(source_lineage)
            self.assertEqual(lineages[0], set(range(rank_size)))
            self.assertEqual(len(lineages[0]), rank_size)
