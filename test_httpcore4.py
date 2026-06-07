import asyncio
import httpx
import httpcore

class SafeNetworkBackend(httpcore.AsyncNetworkBackend):
    def __init__(self, original, safe_ip):
        self.original = original
        self.safe_ip = safe_ip

    async def connect_tcp(self, host, port, timeout, local_address=None, **kwargs):
        print(f"Intercepting connect_tcp: {host} -> {self.safe_ip}")
        # Always dial safe_ip
        return await self.original.connect_tcp(
            self.safe_ip, port, timeout, local_address=local_address, **kwargs
        )

    async def connect_unix_socket(self, *args, **kwargs):
        return await self.original.connect_unix_socket(*args, **kwargs)

    async def sleep(self, *args, **kwargs):
        return await self.original.sleep(*args, **kwargs)

async def main():
    transport = httpx.AsyncHTTPTransport()
    # Mock safe IP as example.com's IP (93.184.215.14)
    original_backend = transport._pool._network_backend
    transport._pool._network_backend = SafeNetworkBackend(original_backend, "93.184.215.14")
    
    async with httpx.AsyncClient(transport=transport) as client:
        # We request an arbitrary URL that resolves to something else normally
        # but because of the override, it should hit example.com.
        # SNI will be 'example.org' which example.com might reject or accept depending on CDN.
        # But for plain HTTP:
        r = await client.get("http://neverssl.com/")
        print("Status:", r.status_code)
        print("Content-Length:", len(r.text))
        print("Headers:", r.headers)

asyncio.run(main())
