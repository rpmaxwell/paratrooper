"""The precomputed intercept tables and the p(hit) matrix agree with brute
force simulation against the game's hitboxes. Full checks: python3
tests/check_tables.py 3000, python3 tests/check_prob_plan.py 3000."""
import check_prob_plan
import check_tables


def test_tables_match_brute_force():
    assert check_tables.main(300)


def test_prob_plan_matches_brute_force():
    assert check_prob_plan.main(300)
