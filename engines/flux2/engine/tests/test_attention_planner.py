from flux2_engine.attention import analyze_attention_candidates


def test_attention_growth_exposes_quadratic_score_cost():
    at_512 = analyze_attention_candidates(width=512, height=512, text_tokens=21)
    at_1024 = analyze_attention_candidates(width=1024, height=1024, text_tokens=21)
    full_512 = next(item for item in at_512 if item.name == "ane-full-attention+mlx-mlp")
    full_1024 = next(item for item in at_1024 if item.name == "ane-full-attention+mlx-mlp")
    assert full_1024.materialized_score_bytes > full_512.materialized_score_bytes * 15
    assert full_1024.status == "rejected-measured"
    assert at_512[0].status == "production"
