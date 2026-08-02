import itertools
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from algorithms.con_method.offline_library import (
    ConLibrary,
    LibraryPathCandidate,
    LibraryScheduleTemplate,
    OfflineLibraryScenario,
    ScenarioConfiguration,
    TopologyLibraryProblem,
    build_waxman_topology_pool,
    build_deterministic_scenario_configurations,
    compile_topology_library,
    instantiate_con_library_for_episode,
    make_waxman_pool_problem_for_slot,
    solve_topology_schedule_library,
    waxman_topology_fingerprint,
)
from qnet_core.contracts.complete_schedule import (
    complete_schedule_count,
    enumerate_complete_schedules,
)
from qnet_core.order_core import OrderLinkSpec
from qnet_core.order_waxman import (
    WaxmanOrderConfig,
    WaxmanOrderEpisode,
    WaxmanOrderRequest,
)


class TopologyCompilerTests(unittest.TestCase):
    @staticmethod
    def _episode():
        nodes = (0, 1, 2, 3, 4)
        edges = ((0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (3, 4))
        config = WaxmanOrderConfig(
            node_count=5,
            average_degree=2,
            request_count=1,
            arrival_rate=1.0,
            episode_steps=1,
            request_ttl_slots=1,
            min_hops=1,
            max_hops=4,
            candidate_paths=4,
            order_variants_per_path=4,
            node_memory_cap=4,
            swap_probability=1.0,
        )
        return WaxmanOrderEpisode(
            seed=0,
            config=config,
            nodes=nodes,
            links=tuple(
                OrderLinkSpec(
                    left,
                    right,
                    capacity=1 + (index % 2),
                    generation_probability=0.5 + index / 20.0,
                )
                for index, (left, right) in enumerate(edges)
            ),
            node_capacities=tuple((node, 2 + node % 3) for node in nodes),
            positions=tuple((node, (float(node), 0.0)) for node in nodes),
            requests=(),
            request_paths=(),
            topology_beta=1.0,
            link_alpha=1.0,
            horizon_slots=1,
        )

    def _compiled(self):
        episode = self._episode()
        pool = build_waxman_topology_pool(
            episode,
            path_pool_per_pair=4,
            max_hops=4,
        )
        return episode, pool, compile_topology_library(pool).library

    def test_compiles_every_unordered_pair_with_bounded_grid(self):
        episode, pool, library = self._compiled()

        expected_pairs = set(itertools.combinations(episode.nodes, 2))
        self.assertEqual(
            {endpoints for _, endpoints in pool.pair_entries},
            expected_pairs,
        )
        self.assertEqual(len(pool.pair_entries), 10)
        self.assertEqual(len(library.pair_entries), 10)
        self.assertEqual(episode.request_paths, ())

        topology_edges = {
            frozenset((link.left, link.right)) for link in episode.links
        }
        for endpoints in expected_pairs:
            grid = library.lookup_grid(*endpoints)
            self.assertEqual(len(grid.candidates), 16)
            self.assertLessEqual(sum(grid.valid_mask), 16)
            valid_paths = []
            for path_slot, row_mask in enumerate(grid.schedule_valid_mask):
                row_count = sum(row_mask)
                if row_count == 0:
                    continue
                candidate = grid.resolve(path_slot, 0)
                path = candidate.path
                valid_paths.append(path)
                self.assertEqual((path[0], path[-1]), endpoints)
                self.assertEqual(len(path), len(set(path)))
                self.assertTrue(all(
                    frozenset(edge) in topology_edges
                    for edge in zip(path, path[1:])
                ))
                self.assertEqual(
                    row_count,
                    min(4, complete_schedule_count(len(path) - 2)),
                )
                schedules = tuple(
                    grid.resolve(path_slot, schedule_slot).schedule
                    for schedule_slot in range(row_count)
                )
                self.assertEqual(
                    len({schedule.structural_key for schedule in schedules}),
                    len(schedules),
                )
            self.assertEqual(len(valid_paths), len(set(valid_paths)))
            self.assertLessEqual(len(valid_paths), 4)

        one_to_four = library.lookup_grid(1, 4)
        self.assertEqual(one_to_four.path_valid_mask, (True, True, True, False))
        self.assertEqual(
            one_to_four.schedule_valid_mask,
            (
                (True, False, False, False),
                (True, True, False, False),
                (True, True, True, True),
                (False, False, False, False),
            ),
        )
        with self.assertRaisesRegex(ValueError, "masked padding"):
            one_to_four.resolve(0, 1)

    def test_reverse_lookup_preserves_slots_groups_and_release_rounds(self):
        _, _, library = self._compiled()
        forward = library.lookup_grid(1, 4)
        reverse = library.lookup_grid(4, 1)

        self.assertEqual(forward.pair_id, reverse.pair_id)
        self.assertEqual(forward.valid_mask, reverse.valid_mask)
        for slot, is_valid in enumerate(forward.valid_mask):
            if not is_valid:
                self.assertIsNone(reverse.candidates[slot])
                continue
            left = forward.candidates[slot]
            right = reverse.candidates[slot]
            self.assertIsNotNone(left)
            self.assertIsNotNone(right)
            self.assertEqual(left.template_id, right.template_id)
            self.assertEqual(left.path_id, right.path_id)
            self.assertEqual(right.path, tuple(reversed(left.path)))
            self.assertEqual(
                tuple(map(frozenset, left.groups)),
                tuple(map(frozenset, right.groups)),
            )
            self.assertEqual(
                left.schedule.release_round_by_node,
                right.schedule.release_round_by_node,
            )
        self.assertEqual(
            library.lookup_grid(1, 4),
            forward,
        )

    def test_cache_roundtrip_and_canonical_topology_validation(self):
        episode, _, library = self._compiled()
        reordered = WaxmanOrderEpisode(
            seed=episode.seed,
            config=episode.config,
            nodes=tuple(reversed(episode.nodes)),
            links=tuple(reversed(episode.links)),
            node_capacities=tuple(reversed(episode.node_capacities)),
            positions=episode.positions,
            requests=episode.requests,
            request_paths=episode.request_paths,
            topology_beta=episode.topology_beta,
            link_alpha=episode.link_alpha,
            horizon_slots=episode.horizon_slots,
        )
        self.assertEqual(
            waxman_topology_fingerprint(episode),
            waxman_topology_fingerprint(reordered),
        )

        with TemporaryDirectory() as directory:
            path = Path(directory) / "con-library.json"
            library.save(path)
            first_bytes = path.read_bytes()
            loaded = ConLibrary.load(
                path,
                expected_topology_fingerprint=waxman_topology_fingerprint(episode),
                expected_compiler_fingerprint=library.compiler_fingerprint,
            )
            self.assertEqual(loaded, library)
            loaded.save(path)
            self.assertEqual(path.read_bytes(), first_bytes)
            self.assertEqual(
                loaded.lookup_grid(4, 1),
                library.lookup_grid(4, 1),
            )
            with self.assertRaisesRegex(ValueError, "topology fingerprint"):
                ConLibrary.load(path, expected_topology_fingerprint="wrong")

    def test_artifact_replaces_request_local_paths_for_online_episode(self):
        episode, _, library = self._compiled()
        request_episode = replace(
            episode,
            requests=(WaxmanOrderRequest("r", 4, 1, 0, 1, 2),),
            request_paths=(("r", ((4, 3, 1),)),),
        )
        installed = instantiate_con_library_for_episode(
            request_episode, library
        )
        grid = library.lookup_grid(4, 1)
        expected_paths = tuple(
            grid.resolve(path_slot, 0).path
            for path_slot, valid in enumerate(grid.path_valid_mask)
            if valid
        )

        self.assertEqual(installed.paths["r"], expected_paths)
        self.assertEqual(
            installed.schedule_library_source,
            "con-offline-artifact-v1",
        )
        self.assertEqual(installed.schedule_library_digest, library.layout_digest)
        problem = installed.problem_for_slot(("r",), 0, physics_seed=0)
        self.assertEqual(len(problem.candidates), sum(grid.valid_mask))

    def test_full_pool_can_build_nominal_training_scenario_columns(self):
        episode, pool, _ = self._compiled()
        request_episode = replace(
            episode,
            requests=(WaxmanOrderRequest("r", 4, 1, 0, 1, 2),),
            request_paths=(("r", ((4, 3, 1),)),),
        )
        slot_problem = make_waxman_pool_problem_for_slot(
            request_episode,
            pool,
            ("r",),
            0,
        )
        pair_id = pool.pair_id_by_endpoints[(1, 4)]
        expected_templates = sum(
            len(pool.templates_by_path[path.path_id])
            for path in pool.paths_by_pair[pair_id]
        )

        self.assertEqual(len(slot_problem.problem.candidates), expected_templates)
        self.assertEqual(
            set(slot_problem.template_id_by_plan_id.values()),
            {
                template.template_id
                for path in pool.paths_by_pair[pair_id]
                for template in pool.templates_by_path[path.path_id]
            },
        )
        self.assertEqual(slot_problem.problem.config.swap_probability, 1.0)
        self.assertTrue(all(
            link.generation_probability == 1.0
            for link in slot_problem.problem.links
        ))
        columns = build_deterministic_scenario_configurations(
            slot_problem.problem,
            slot_problem.template_id_by_plan_id,
        )
        self.assertEqual(columns.enumerated_assignments, expected_templates + 1)
        self.assertGreater(columns.feasible_assignments, 1)

    def test_joint_path_schedule_milp_matches_independent_bruteforce(self):
        weights = (
            (10, 9, 8, 7, 1),
            (9, 8, 7, 6, 1),
            (8, 7, 6, 5, 1),
            (7, 6, 5, 4, 1),
            (6, 5, 4, 3, 1),
        )
        paths = []
        templates = []
        scenarios = []
        for path_index in range(5):
            path = (
                "S",
                f"{path_index}A",
                f"{path_index}B",
                f"{path_index}C",
                "T",
            )
            path_id = f"p{path_index}"
            paths.append(LibraryPathCandidate(
                pair_id="pair",
                path_id=path_id,
                path=path,
                pool_rank=path_index,
            ))
            for schedule_index, schedule in enumerate(
                enumerate_complete_schedules(path)[:5]
            ):
                template_id = f"{path_id}:s{schedule_index}"
                templates.append(LibraryScheduleTemplate(
                    template_id=template_id,
                    path_id=path_id,
                    schedule=schedule,
                    pair_id="pair",
                ))
                scenarios.append(OfflineLibraryScenario(
                    scenario_id=f"single:{path_index}:{schedule_index}",
                    trace_digest=f"trace:{path_index}:{schedule_index}",
                    topology_fingerprint="topology",
                    request_distribution_fingerprint="requests",
                    physics_fingerprint="physics",
                    configurations=(
                        ScenarioConfiguration("empty"),
                        ScenarioConfiguration(
                            "serve",
                            frozenset({template_id}),
                            frozenset({f"r:{path_index}:{schedule_index}"}),
                        ),
                    ),
                    weight=weights[path_index][schedule_index],
                ))
        scenarios.append(OfflineLibraryScenario(
            scenario_id="synergy",
            trace_digest="trace:synergy",
            topology_fingerprint="topology",
            request_distribution_fingerprint="requests",
            physics_fingerprint="physics",
            configurations=(
                ScenarioConfiguration("empty"),
                ScenarioConfiguration(
                    "serve",
                    frozenset({"p3:s4", "p4:s4"}),
                    frozenset({"synergy-request"}),
                ),
            ),
            weight=50,
        ))

        brute_best = -1
        brute_argmax = []
        for selected_path_indices in itertools.combinations(range(5), 4):
            for schedule_choices in itertools.product(
                itertools.combinations(range(5), 4), repeat=4
            ):
                selected = frozenset(
                    f"p{path_index}:s{schedule_index}"
                    for path_index, schedules in zip(
                        selected_path_indices, schedule_choices
                    )
                    for schedule_index in schedules
                )
                score = sum(
                    weights[path_index][schedule_index]
                    for path_index in range(5)
                    for schedule_index in range(5)
                    if f"p{path_index}:s{schedule_index}" in selected
                )
                if {"p3:s4", "p4:s4"} <= selected:
                    score += 50
                if score > brute_best:
                    brute_best = score
                    brute_argmax = [selected]
                elif score == brute_best:
                    brute_argmax.append(selected)

        result = solve_topology_schedule_library(TopologyLibraryProblem(
            paths=tuple(paths),
            templates=tuple(templates),
            scenarios=tuple(scenarios),
            paths_per_pair=4,
            schedules_per_path=4,
        ))
        selected = frozenset(result.library.selected_template_ids)

        self.assertEqual(brute_best, 149)
        self.assertEqual(result.training_weighted_completed, 149)
        self.assertEqual(result.training_total_completed, 17)
        self.assertEqual(result.solver_mip_gap, 0.0)
        self.assertIn(selected, brute_argmax)
        self.assertEqual(
            set(result.library.selected_by_pair["pair"]),
            {"p0", "p1", "p3", "p4"},
        )
        self.assertFalse(any(
            template_id in selected for template_id in (
                f"p2:s{index}" for index in range(5)
            )
        ))
        self.assertTrue({"p3:s4", "p4:s4"} <= selected)
        self.assertEqual(
            {
                path_id: len(template_ids)
                for path_id, template_ids
                in result.library.selected_by_path.items()
                if template_ids
            },
            {"p0": 4, "p1": 4, "p3": 4, "p4": 4},
        )


if __name__ == "__main__":
    unittest.main()
