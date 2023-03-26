import asyncio
import numpy as np

async def get_number():
    return np.random.rand()

async def main():
    # Schedule three calls *concurrently*:
    coros = [
        get_number(),
        get_number(),
        get_number(),
        get_number()
    ]
    L = await asyncio.gather(*coros)
    print(L)

asyncio.run(main())
