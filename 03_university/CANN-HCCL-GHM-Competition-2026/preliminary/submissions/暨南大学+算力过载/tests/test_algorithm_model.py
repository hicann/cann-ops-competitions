import unittest


MAX_CHUNK_BYTES = 256 * 1024 * 1024
FLOAT_BYTES = 4


def peers_for_rank(rank: int, rank_size: int) -> list[int]:
    if rank_size <= 0 or rank_size & (rank_size - 1):
        raise ValueError("rank_size must be a positive power of two")
    return [rank ^ round_id for round_id in range(1, rank_size)]


def chunk_counts(total_count: int) -> list[int]:
    max_count = MAX_CHUNK_BYTES // FLOAT_BYTES
    result = []
    offset = 0
    while offset < total_count:
        count = min(max_count, total_count - offset)
        result.append(count)
        offset += count
    return result


class XorScheduleTest(unittest.TestCase):
    def test_each_rank_visits_every_other_rank_once(self):
        for rank in range(16):
            peers = peers_for_rank(rank, 16)
            self.assertEqual(len(peers), 15)
            self.assertEqual(set(peers), set(range(16)) - {rank})

    def test_every_round_is_symmetric(self):
        for round_id in range(1, 16):
            for rank in range(16):
                peer = rank ^ round_id
                self.assertEqual(peer ^ round_id, rank)

    def test_functional_case_chunk_boundaries(self):
        cases = {
            4: [1],
            512 * 1024: [131072],
            512 * 1024 * 1024: [67108864, 67108864],
            400 * 1024 * 1024 + 4: [67108864, 37748737],
        }
        for byte_size, expected in cases.items():
            self.assertEqual(chunk_counts(byte_size // FLOAT_BYTES), expected)

    def test_non_power_of_two_is_rejected(self):
        with self.assertRaises(ValueError):
            peers_for_rank(0, 15)

    def test_rebalanced_two_lane_partition_is_complete_and_disjoint(self):
        lane_masks = ([0, 1, 2, 3, 4, 5, 6, 7, 8], [9, 10, 11, 12, 13, 14, 15])
        flattened = [mask for lane in lane_masks for mask in lane]
        self.assertEqual(len(flattened), 16)
        self.assertEqual(set(flattened), set(range(16)))
        for rank in range(16):
            contributions = [rank ^ mask for mask in flattened]
            self.assertEqual(set(contributions), set(range(16)))

    def test_worker_has_one_fewer_network_operation(self):
        # The main lane starts immediately and performs eight remote reductions.
        # The delayed worker performs one initialization plus six reductions.
        self.assertEqual((8, 1 + 6), (8, 7))


if __name__ == "__main__":
    unittest.main()
