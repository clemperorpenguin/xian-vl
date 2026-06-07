import httpx

t = httpx.AsyncHTTPTransport()
print(dir(t))
print(type(t._pool))
