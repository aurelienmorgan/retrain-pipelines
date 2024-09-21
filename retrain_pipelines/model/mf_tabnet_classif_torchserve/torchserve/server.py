
import time
import requests

def await_server_ready(
    retries: int = 10,
    delay: int = 2
) -> bool:
    """
    Params:
        - retries (int)
        - delay (int):
            in seconds
    """

    url = "http://localhost:9080/ping"
    for i in range(retries):
        try:
            response = requests.get(url)
            if response.status_code == 200:
                print("Server is ready!")
                return True
        except requests.RequestException as e:
            print(f"Request failed: {e}")
        
        print(f"Retrying in {delay} seconds...")
        time.sleep(delay)
    
    print(f"Server did not become ready after {retries*delay} seconds.")

    return False

