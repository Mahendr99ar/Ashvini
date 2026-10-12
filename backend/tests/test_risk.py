

def test_delta_models_add_their_change_to_the_current_risk():
    from ashvini.risk.model import combine

    assert combine("delta_from_risk", -5.0, 0.5, 0.8) == -4.6
    assert combine("risk", -5.0, -4.2, 0.8) == -4.2
