import requests

def test_app():
    url = "http://127.0.0.1:5000/predict"
    data = {"tweet": "I feel very sad and hopeless today"}
    try:
        response = requests.post(url, data=data)
        if response.status_code == 200:
            print("Successfully connected to the server.")
            if "Outcome: Depressive" in response.text:
                print("Prediction test PASSED: Correctly identified depressive sentiment.")
            else:
                print("Prediction test FAILED: Outcome not found in response.")
        else:
            print(f"Server returned status code: {response.status_code}")
    except Exception as e:
        print(f"Error connecting to server: {e}")

if __name__ == "__main__":
    test_app()
