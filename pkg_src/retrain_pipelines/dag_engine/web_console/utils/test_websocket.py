
import json
import socket
import asyncio
import websockets


##############################


# testing raw websocket connectivity
s = socket.socket()
try:
    s.connect(("0.0.0.0", 5001))
    print("✅ TCP connection succeeded.")
except Exception as e:
    print("❌ TCP connection failed:", e)
finally:
    s.close()


##############################


async def test_ws():
    uri = "ws://localhost:5001/ws/test"
    try:
        async with websockets.connect(uri) as websocket:
            print("✅ Connected to", uri)

            payload = json.dumps({"message": "ping"})
            await websocket.send(payload)
            print(f"📨 Sent: {payload}")

            try:
                async for message in websocket:
                    print("📩 Received:", message)
            except websockets.exceptions.ConnectionClosed:
                print("🔚 Connection closed")

    except Exception as e:
        print("❌ Error:", e)

asyncio.run(test_ws())


##############################

