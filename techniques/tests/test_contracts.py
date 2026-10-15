"""Local core-design ownership, malformed pack, refill and policy controls."""
import copy
import json
from pathlib import Path
import struct
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "labs"))
import contracts


class GraphOwnership(unittest.TestCase):
    def test_snapshot_seal_and_copy_rejection(self):
        graph, owner, params = contracts.OwnedGraph(), [], {"nested": [3]}
        graph.add(lambda owners, p: owners[0].extend(p["nested"]), [owner], params)
        for copier in (copy.copy, copy.deepcopy):
            with self.assertRaises(TypeError):
                copier(graph)
        ticket = graph.submit()
        params["nested"][0] = 99
        for operation in (graph.submit, lambda: graph.add(lambda *_: None, [], {})):
            with self.assertRaises(ValueError):
                operation()
        self.assertTrue(ticket.finish())
        self.assertEqual(owner, [3])
        with self.assertRaises(ValueError):
            ticket.finish()

    def test_cancel_is_not_rollback_or_permission_to_release(self):
        graph, output = contracts.OwnedGraph(), []
        graph.add(lambda owners, _: owners[0].append(1), [output], {})
        ticket = graph.submit()
        del graph
        ticket.cancel_or_timeout()
        self.assertTrue(ticket.commands)
        self.assertFalse(ticket.physically_complete)
        self.assertFalse(ticket.finish())
        self.assertEqual(output, [1])
        self.assertFalse(ticket.commands)

    def test_failed_operation_clears_owners_without_success(self):
        def failed(*_):
            raise RuntimeError("producer failed")
        graph = contracts.OwnedGraph()
        graph.add(failed, [object()], {})
        ticket = graph.submit()
        with self.assertRaisesRegex(RuntimeError, "producer failed"):
            ticket.finish()
        self.assertTrue(ticket.physically_complete)
        self.assertFalse(ticket.succeeded)
        self.assertFalse(ticket.commands)


class BankLifetimes(unittest.TestCase):
    def test_malformed_token_and_ambiguous_completion_rejected(self):
        pool = contracts.DualBank()
        token = pool.claim("A")
        for invalid in ((0, True), (0,), "A", (5, 1)):
            with self.assertRaises(ValueError):
                pool.stage_complete(invalid)
        with self.assertRaises(ValueError):
            pool.stage_complete(token, success="failed")
        pool.stage_complete(token)
        pool.launch(token, "A")
        with self.assertRaises(ValueError):
            pool.physical_complete(token, success=1)

    def test_ready_source_and_backpressure(self):
        pool = contracts.DualBank()
        a, b = pool.claim("A"), pool.claim("B")
        with self.assertRaises(ValueError):
            pool.claim("C")
        with self.assertRaises(ValueError):
            pool.launch(a, "A")
        pool.stage_complete(a)
        with self.assertRaises(ValueError):
            pool.launch(a, "B")
        pool.launch(a, "A")
        pool.timeout(a)
        with self.assertRaises(ValueError):
            pool.claim("C")
        self.assertFalse(pool.physical_complete(a))
        c = pool.claim("C")
        self.assertNotEqual(a, c)
        with self.assertRaises(ValueError):
            pool.physical_complete(a)
        pool.stage_complete(b, success=False)
        with self.assertRaises(ValueError):
            pool.launch(b, "B")
        pool.stage_complete(c)
        pool.launch(c, "C")
        self.assertTrue(pool.physical_complete(c))

    def test_failed_execution_is_not_publishable(self):
        pool = contracts.DualBank()
        token = pool.claim("A")
        pool.stage_complete(token)
        pool.launch(token, "A")
        self.assertFalse(pool.physical_complete(token, success=False))


class PackAndRefill(unittest.TestCase):
    def test_invalid_schema_and_duplicate_json_keys(self):
        for header in ([], {}, {"version": True, "endian": "little", "tensors": []},
                       {"version": 1, "endian": "little", "tensors": [None]},
                       {"version": 1, "endian": "little", "tensors": [{}]}):
            encoded = json.dumps(header).encode()
            raw = contracts.MAGIC + struct.pack("<I", len(encoded)) + encoded
            with self.assertRaises(ValueError):
                contracts.PackedStore(raw)
        encoded = b'{"version":1,"version":1,"endian":"little","tensors":[]}'
        with self.assertRaisesRegex(ValueError, "duplicate pack metadata key"):
            contracts.PackedStore(contracts.MAGIC + struct.pack("<I", len(encoded)) + encoded)

    def fixture(self):
        return contracts.make_pack([("a", (2,), "F16", struct.pack("<2e", 1., -.5)),
                                    ("b", (1,), "F32", struct.pack("<f", .25))])

    def rewrite_header(self, raw, change):
        size = struct.unpack_from("<I", raw, 8)[0]
        old_payload = raw[contracts.aligned(12 + size):]
        header = json.loads(raw[12:12 + size])
        change(header)
        encoded = json.dumps(header).encode()
        prefix = contracts.MAGIC + struct.pack("<I", len(encoded)) + encoded
        return prefix + b"\0" * (contracts.aligned(len(prefix)) - len(prefix)) + old_payload

    def test_readonly_backing_survives_store_and_metadata_cannot_mutate(self):
        store = contracts.PackedStore(self.fixture())
        view = store.view("a")
        with self.assertRaises(TypeError):
            view[0] = 0
        with self.assertRaises(TypeError):
            store.entries["a"]["offset"] = 64
        del store
        self.assertEqual(struct.unpack("<2e", view), (1., -.5))

    def test_malformed_header_and_payload_rejected(self):
        raw = self.fixture()
        changes = (
            lambda h: h.update(version=2), lambda h: h.update(endian="big"),
            lambda h: h["tensors"][0].update(offset=1),
            lambda h: h["tensors"][0].update(offset=1 << 30),
            lambda h: h["tensors"][1].update(offset=0),
            lambda h: h["tensors"][1].update(name="a"),
            lambda h: h["tensors"][0].update(shape=[True]),
            lambda h: h["tensors"][0].update(shape=[3]),
            lambda h: h["tensors"][0].update(dtype="Q4"),
        )
        for change in changes:
            with self.assertRaises(ValueError):
                contracts.PackedStore(self.rewrite_header(raw, change))
        for damaged in (b"bad", b"badmagic" + raw[8:], raw[:10], raw[:-1]):
            with self.assertRaises(ValueError):
                contracts.PackedStore(damaged)

    def test_f16_to_bf16_is_numeric_not_reinterpret(self):
        values = [1., -.5, 2., .25, 1.00390625]
        source = struct.pack("<5e", *values)
        result = contracts.typed_refill(source, 5, "F16", "BF16")
        self.assertEqual(result["status"], "ready")
        actual = [struct.unpack("<f", b"\0\0" + result["payload"][i:i + 2])[0] for i in range(0, 10, 2)]
        self.assertEqual(actual, [1., -.5, 2., .25, 1.])  # BF16 midpoint, ties to even.
        self.assertNotEqual(source, result["payload"])
        self.assertEqual(contracts.typed_refill(result["payload"], 5, "BF16", "BF16")["payload"], result["payload"])

    def test_short_invalid_conversion_never_publishes_partial(self):
        for raw in (b"\0", struct.pack("<e", float("inf")), struct.pack("<e", float("nan"))):
            result = contracts.typed_refill(raw, 1, "F16", "BF16")
            self.assertEqual(result, {"status": "failed", "payload": None})
        self.assertEqual(contracts.typed_refill(b"", 1, "Q4", "F16"), {"status": "unservable", "payload": None})
        self.assertEqual(contracts.typed_refill(struct.pack("<f", 1.), 1, "F32", "BF16")["status"], "unservable")


class ResidencyPolicy(unittest.TestCase):
    def test_unknown_reserve_and_hysteresis(self):
        policy = contracts.FixedResidency()
        self.assertFalse(policy.admit("A", 10, 1000))
        policy.reserve = 100
        self.assertFalse(policy.admit("A", 10, 119))
        self.assertTrue(policy.admit("A", 10, 120))
        before = dict(policy.blocks)
        self.assertFalse(policy.admit("B", 100, 100))
        self.assertEqual(policy.blocks, before)  # Admission never evicts.

    def test_reserve_subtraction_ratchet_and_explicit_probe(self):
        details = contracts.residency_policy()["details"]
        self.assertEqual(details["lru_warm_hits"], 0)
        self.assertEqual(details["fixed_warm_hits"], 6)
        self.assertTrue(details["allocated_reserve_not_counted_twice"])
        self.assertTrue(details["post_eviction_ratchet"])
        self.assertTrue(details["explicit_healthy_probe_can_reopen"])

    def test_invalid_byte_budgets(self):
        for reserve in (-1, True, float("nan")):
            with self.assertRaises(ValueError):
                contracts.FixedResidency(reserve)
        policy = contracts.FixedResidency(0)
        for size, available in ((0, 100), (True, 100), (10, -1), (10, float("nan"))):
            with self.assertRaises(ValueError):
                policy.admit("A", size, available)


class RuntimeProfit(unittest.TestCase):
    def test_warmup_window_sticky_disable_and_requalification(self):
        details = contracts.hybrid_policy()["details"]
        self.assertTrue(details["sustained_regression_disables"])
        self.assertTrue(details["no_implicit_reenable"])
        self.assertTrue(details["explicit_requalification_can_reopen"])

    def test_noisy_single_slow_sample_does_not_trip_median(self):
        breaker = contracts.ProfitBreaker("A", warmup=0)
        for duration in (.008, .008, .02, .008):
            self.assertTrue(breaker.observe("A", .01, duration))
        self.assertEqual(len(breaker.ratios), 4)

    def test_changed_identity_invalid_timings_and_configuration_rejected(self):
        breaker = contracts.ProfitBreaker("A", warmup=0)
        for identity, gpu, split in (("B", .01, .01), ("A", 0, .01), ("A", True, .01), ("A", .01, float("nan"))):
            with self.assertRaises(ValueError):
                breaker.observe(identity, gpu, split)
        self.assertFalse(breaker.ratios)
        for config in ({"window": 1}, {"warmup": -1}, {"maximum_ratio": float("nan")}):
            with self.assertRaises(ValueError):
                contracts.ProfitBreaker("A", **config)


if __name__ == "__main__":
    unittest.main()
