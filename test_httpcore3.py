import httpx

t = httpx.AsyncHTTPTransport()
print(dir(t._pool))
if hasattr(t._pool, '_network_backend'):
    print(t._pool._network_backend)
