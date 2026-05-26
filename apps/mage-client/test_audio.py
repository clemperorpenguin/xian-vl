import asyncio
import struct
import math
import subprocess

async def test():
    proc = await asyncio.create_subprocess_exec(
        "parecord", "--rate", "16000", "--channels", "1", "--format", "s16le", "--device", "@DEFAULT_MONITOR@", "-",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    print("Reading 5 chunks of 100ms...")
    for i in range(50):
        try:
            data = await asyncio.wait_for(proc.stdout.readexactly(3200), timeout=1.0)
            samples = struct.unpack(f'<{len(data)//2}h', data)
            rms = math.sqrt(sum(s*s for s in samples) / len(samples))
            print(f"Chunk {i}: RMS = {rms:.2f}, max = {max(samples)}, min = {min(samples)}")
        except Exception as e:
            print("Error reading:", e)
            break
            
    proc.terminate()
    
asyncio.run(test())
