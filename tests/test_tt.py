import unittest

from src.tt import BOUND_EXACT, BOUND_LOWER, create_tt_arrays, probe_tt, store_tt


class TranspositionTableTests(unittest.TestCase):
    def setUp(self) -> None:
        self.arrays = create_tt_arrays(1)

    def _store(self, hash_value: int, depth: int, bound: int, age: int) -> None:
        store_tt(hash_value, 1, 25, depth, bound, 0, age, *self.arrays)

    def test_collision_preserves_deeper_current_entry(self) -> None:
        self._store(0, 8, BOUND_EXACT, 1)
        self._store(2, 3, BOUND_LOWER, 1)

        found, _, _, depth, bound, _ = probe_tt(0, *self.arrays)

        self.assertTrue(found)
        self.assertEqual(depth, 8)
        self.assertEqual(bound, BOUND_EXACT)

    def test_shallow_exact_entry_does_not_replace_deeper_exact_entry(self) -> None:
        self._store(0, 8, BOUND_EXACT, 1)
        self._store(0, 3, BOUND_EXACT, 1)

        found, _, _, depth, bound, _ = probe_tt(0, *self.arrays)

        self.assertTrue(found)
        self.assertEqual(depth, 8)
        self.assertEqual(bound, BOUND_EXACT)

    def test_stale_entry_is_replaced_after_collision(self) -> None:
        self._store(0, 8, BOUND_EXACT, 1)
        self._store(2, 3, BOUND_LOWER, 4)

        found, _, _, depth, bound, _ = probe_tt(2, *self.arrays)

        self.assertTrue(found)
        self.assertEqual(depth, 3)
        self.assertEqual(bound, BOUND_LOWER)


if __name__ == "__main__":
    unittest.main()
