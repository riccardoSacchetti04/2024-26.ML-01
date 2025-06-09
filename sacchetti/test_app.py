import pytest
from app import app as flask_app

@pytest.fixture()
def client():
    flask_app.config.update({"TESTING": True})
    with flask_app.test_client() as client:
        yield client

def test_prediction(client):
    sample_input = {
        "Sex": 1,
        "PhysicalActivities": 1,
        "HadAngina": 0,
        "HadStroke": 0,
        "HadAsthma": 0,
        "HadSkinCancer": 0,
        "HadCOPD": 0,
        "HadDepressiveDisorder": 0,
        "HadKidneyDisease": 0,
        "HadArthritis": 0,
        "DeafOrHardOfHearing": 0,
        "BlindOrVisionDifficulty": 0,
        "DifficultyConcentrating": 0,
        "DifficultyWalking": 0,
        "DifficultyDressingBathing": 0,
        "DifficultyErrands": 0,
        "ChestScan": 0,
        "AlcoholDrinkers": 0,
        "HIVTesting": 0,
        "FluVaxLast12": 0,
        "PneumoVaxEver": 0,
        "HighRiskLastYear": 0,
        "CovidPos": "0",
        "GeneralHealth": "2",
        "LastCheckupTime": "0",
        "RemovedTeeth": "0",
        "SmokerStatus": "1",
        "ECigaretteUsage": "0",
        "AgeCategory": "6",
        "State": "California",
        "RaceEthnicityCategory": "White",
        "TetanusLast10Tdap": "Yes",
        "HadDiabetes": "No"
    }

    response = client.post("/infer", json=sample_input)
    assert response.status_code == 200
    data = response.get_json()
    assert "prediction" in data
    assert isinstance(data["prediction"], list)
    assert data["prediction"][0] in [0, 1]
