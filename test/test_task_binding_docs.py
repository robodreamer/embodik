import embodik as eik


def test_task_weight_docstring_names_position_step_gain_surface():
    doc = eik.Task.weight.__doc__

    assert doc is not None
    assert "velocity-IK path only" in doc
    assert "solve_position_step()" in doc
    assert "TaskTarget position_gain/orientation_gain" in doc
