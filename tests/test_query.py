import unittest

from glance_retrieval.query import binding_components, binding_distractors, parse_query


class QueryParserTests(unittest.TestCase):
    def test_binds_color_to_garment_in_compositional_query(self):
        parsed = parse_query("A red tie and a white shirt in a formal setting.")
        self.assertEqual(parsed.bindings, ("red tie", "white shirt"))
        self.assertIn("formal setting", parsed.scenes)
        self.assertIn("formal outfit", parsed.styles)


    def test_context_action_and_binding_are_separate(self):
        parsed = parse_query("Someone wearing a blue shirt sitting on a park bench.")
        self.assertEqual(parsed.bindings, ("blue shirt",))
        self.assertEqual(parsed.scenes, ("park",))
        self.assertEqual(parsed.actions, ("sitting",))


    def test_bright_yellow_raincoat_is_one_binding(self):
        parsed = parse_query("A person in a bright yellow raincoat.")
        self.assertEqual(parsed.bindings, ("bright yellow raincoat",))

    def test_binding_counterfactuals_include_wrong_garments_and_swapped_color(self):
        self.assertEqual(binding_components("bright yellow raincoat"), ("yellow", "raincoat"))
        distractors = binding_distractors("red tie", ("red", "white"))
        self.assertIn("red shirt", distractors)
        self.assertIn("red pants", distractors)
        self.assertIn("white tie", distractors)
