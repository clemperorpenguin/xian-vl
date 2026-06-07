import asyncio
import httpcore
import httpx

class SafeNetworkBackend(httpcore.AsyncNetworkBackend):
    def __init__(self, original, safe_ip):
        self.original = original
        self.safe_ip = safe_ip

    async def connect_tcp(self, host, port, timeout, local_address=None, **kwargs):
        print(f"Intercepting connect_tcp: {host} -> {self.safe_ip}")
        return await self.original.connect_tcp(
            self.safe_ip, port, timeout, local_address=local_address, **kwargs
        )

    async def connect_unix_socket(self, *args, **kwargs):
        return await self.original.connect_unix_socket(*args, **kwargs)

    async def sleep(self, *args, **kwargs):
        return await self.original.sleep(*args, **kwargs)

async def main():
    try:
        from httpcore.backends.auto import AutoBackend
        backend = AutoBackend()
    except ImportError:
        try:
            from httpcore._backends.auto import AutoBackend
            backend = AutoBackend()
        except ImportError as e:
            print(f"Error importing backend: {e}")
            return
            
    transport = httpx.AsyncHTTPTransport()
    # In recent httpx/httpcore, transport might have a _pool or we can pass network_backend to HTTPTransport
    
    # Actually, httpx.AsyncHTTPTransport takes `network_backend`? No, httpx 0.28 doesn't expose it.
    pass

asyncio.run(main())
