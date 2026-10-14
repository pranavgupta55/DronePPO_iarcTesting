"""The hand-written controller should fly a short course without crashing (it is the teacher for the imitation runs)."""
from drone_rl.evaluation import FIXED_COURSES, course_points, run_episode
from drone_rl.env import make_spec
from drone_rl.policies import ControllerPolicy


class _Loader:
    def __call__(self):
        return ControllerPolicy()


def test_controller_flies_precision_course():
    spec = make_spec({"scheme": "time_shaping", "params": {}, "kind": "controller", "force_rate": 0.5, "mirror": False})
    result = run_episode((spec, _Loader(), ("fixed:close_precision", 0), 1, 0.0, False))
    assert not result["crashed"]
    assert sum(leg["reached"] for leg in result["legs"]) >= 6


def test_courses_stay_inside_the_world():
    for world in ((1200, 800), (2400, 1600)):
        for name in FIXED_COURSES:
            for x, y in course_points("fixed:" + name, world, 0):
                assert 0 <= x <= world[0] and 0 <= y <= world[1]
        for x, y in course_points("random", world, 7):
            assert 0 <= x <= world[0] and 0 <= y <= world[1]
