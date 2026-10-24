from src.predict import predict_text

class FakeModel:
    def predict(self, texts):
        return ["Example category"]
    def predict_proba(self, texts):
        return [[0.2, 0.8]]

def test_prediction_shape():
    result = predict_text(FakeModel(), "This is a sufficiently long legal clause.")
    assert result["label"] == "Example category"
    assert result["score"] == 0.8
    assert result["elapsed_ms"] >= 0
